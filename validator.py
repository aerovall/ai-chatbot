"""Prop firm validation logic.

This module gates every firm-specific request behind a validation check against
propfirmmatch.com. The check is performed internally and must never be surfaced
to end users: neither the site name nor the fact that validation happened should
appear in user-facing text.

Validation results are cached in the database (``firm_search_cache``) so the
remote site is not scraped repeatedly for the same firm within the refresh
window.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Optional

from database import Database
from scraper import PropFirmScraper

logger = logging.getLogger("tickshift.validator")


# A pool of natural, varied responses used when a firm is not recognised.
# These deliberately never reference the validation source.
UNRECOGNISED_FIRM_MESSAGES: List[str] = [
    "That prop firm isn't well known in the industry.",
    "I don't have reliable information about that firm.",
    "That firm doesn't seem to be established enough for me to provide details.",
    "I'm not familiar with that particular firm.",
    "That doesn't appear to be a recognized prop trading firm.",
    "I couldn't find any dependable information on that firm.",
    "That firm isn't one I can speak to with any confidence.",
    "I don't have solid data on that firm, so I'd rather not guess.",
]


@dataclass
class ValidationResult:
    """Outcome of validating a firm name.

    Attributes:
        firm_name: The name that was validated.
        is_valid: Whether the firm is recognised (listed on the source).
        in_database: Whether the firm now exists in the local database.
        message: A user-facing message to show when ``is_valid`` is ``False``.
    """

    firm_name: str
    is_valid: bool
    in_database: bool = False
    message: Optional[str] = None


class FirmValidator:
    """Validates firms and lazily ingests newly discovered ones into the DB."""

    def __init__(self, db: Database, scraper: PropFirmScraper) -> None:
        """Create the validator.

        Args:
            db: The database helper for cache and firm persistence.
            scraper: The scraper used for validation and data collection.
        """
        self._db = db
        self._scraper = scraper

    @staticmethod
    def random_unrecognised_message() -> str:
        """Return a randomly chosen, natural "firm not recognised" message."""
        return random.choice(UNRECOGNISED_FIRM_MESSAGES)

    def validate_and_ingest(self, firm_name: str) -> ValidationResult:
        """Validate a firm and, if valid, ensure it exists in the database.

        The flow is:

        1. If the firm is already in the local database, treat it as valid
           (subject to cache freshness) and return immediately.
        2. Otherwise consult the search cache. A fresh "not found" result short
           circuits to an unrecognised response without re-scraping.
        3. Otherwise validate against propfirmmatch.com. If listed, scrape its
           data and persist it. If not listed, cache the negative result and
           return an unrecognised message.

        Args:
            firm_name: The firm name to validate and ingest.

        Returns:
            A :class:`ValidationResult` describing the outcome.
        """
        firm_name = firm_name.strip()
        if not firm_name:
            return ValidationResult(
                firm_name=firm_name,
                is_valid=False,
                message=self.random_unrecognised_message(),
            )

        # Step 1: already known locally -> trust it. The local database is the
        # curated knowledge base (seeded catalogue + previously ingested
        # firms), so it is authoritative; external validation exists only to
        # gate firms we know nothing about. Re-validating known firms against
        # the external source caused false "unrecognised" verdicts whenever
        # that site could not be scraped.
        already_local = self._db.firm_exists(firm_name)
        if already_local:
            logger.debug("Firm '%s' served from local DB.", firm_name)
            return ValidationResult(
                firm_name=firm_name, is_valid=True, in_database=True
            )

        # Step 2: consult the cache for a recent negative result.
        cache = self._db.get_cache_entry(firm_name)
        if (
            cache is not None
            and cache["is_valid_firm"] is False
            and self._db.is_cache_fresh(firm_name)
        ):
            logger.debug(
                "Firm '%s' known-invalid from fresh cache; skipping scrape.",
                firm_name,
            )
            return ValidationResult(
                firm_name=firm_name,
                is_valid=False,
                message=self.random_unrecognised_message(),
            )

        # Step 3: perform live validation against the source.
        try:
            listed = self._scraper.is_firm_listed(firm_name)
        except Exception:  # noqa: BLE001 - never let scraping crash a command.
            logger.exception("Validation error for firm '%s'.", firm_name)
            # If we already have local data, fall back to it rather than failing.
            if already_local:
                return ValidationResult(
                    firm_name=firm_name, is_valid=True, in_database=True
                )
            self._safe_update_cache(firm_name, False, "error")
            return ValidationResult(
                firm_name=firm_name,
                is_valid=False,
                message=self.random_unrecognised_message(),
            )

        if not listed:
            logger.info("Firm '%s' is not listed; treating as unrecognised.", firm_name)
            self._safe_update_cache(firm_name, False, "not_found")
            # If it is not recognised, we intentionally do not use any stale
            # local record: the firm failed validation.
            return ValidationResult(
                firm_name=firm_name,
                is_valid=False,
                message=self.random_unrecognised_message(),
            )

        # Firm is valid. Ensure the database has (fresh) data for it.
        in_db = self._ingest_firm(firm_name) or already_local
        self._safe_update_cache(firm_name, True, "found")
        return ValidationResult(
            firm_name=firm_name, is_valid=True, in_database=in_db
        )

    def _ingest_firm(self, firm_name: str) -> bool:
        """Scrape and persist data for a validated firm.

        Args:
            firm_name: The validated firm name.

        Returns:
            ``True`` if the firm now exists in the database.
        """
        try:
            firm_data = self._scraper.scrape_firm(firm_name)
        except Exception:  # noqa: BLE001 - degrade gracefully on scrape errors.
            logger.exception("Failed to scrape data for '%s'.", firm_name)
            firm_data = None

        if firm_data is None:
            # We could validate the firm but not collect data. Persist a stub so
            # future queries still resolve, if it is not already present.
            if not self._db.firm_exists(firm_name):
                try:
                    self._db.upsert_firm({"name": firm_name})
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to persist stub firm '%s'.", firm_name)
                    return False
            return self._db.firm_exists(firm_name)

        try:
            self._db.upsert_firm(firm_data.to_dict())
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist firm '%s'.", firm_name)
            return self._db.firm_exists(firm_name)
        return True

    def _safe_update_cache(
        self, firm_name: str, is_valid: bool, status: str
    ) -> None:
        """Update the search cache, swallowing DB errors (non-critical path)."""
        try:
            self._db.update_cache(firm_name, is_valid, status)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to update search cache for '%s'.", firm_name)
