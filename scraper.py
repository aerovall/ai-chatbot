"""Web scraping utilities for the TickShift Discord bot.

This module is responsible for two things:

1. Checking whether a prop firm is *listed* on propfirmmatch.com. This is used
   internally as a validation gate and must never be surfaced to end users.
2. Collecting structured information about a listed firm (from propfirmmatch.com
   and, where possible, the firm's own website) so it can be persisted to the
   local database.

The scraper is intentionally defensive: the remote site's markup can change at
any time, so every extraction step is best-effort and wrapped in error handling.
It also implements polite behaviour: a shared rate limiter, a descriptive
User-Agent, and robots.txt awareness.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("tickshift.scraper")

USER_AGENT = (
    "TickShiftBot/1.0 (+https://tickshift.ai; prop-firm knowledge base bot)"
)


class RateLimiter:
    """A thread-safe minimum-interval rate limiter.

    Ensures at least ``min_interval`` seconds elapse between successive calls to
    :meth:`wait`, smoothing out request bursts against a remote host.
    """

    def __init__(self, min_interval: float) -> None:
        """Create a rate limiter.

        Args:
            min_interval: Minimum number of seconds between permitted actions.
        """
        self._min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def wait(self) -> None:
        """Block until at least ``min_interval`` has passed since the last call."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            sleep_for = self._min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


@dataclass
class FirmData:
    """Structured data extracted for a single prop firm."""

    name: str
    country: Optional[str] = None
    founded_year: Optional[int] = None
    tier: Optional[str] = None
    max_allocation: Optional[int] = None
    profit_split: Optional[int] = None
    challenge_fee_from: Optional[float] = None
    payout_frequency: Optional[str] = None
    description: Optional[str] = None
    warning_flag: bool = False
    warning_message: Optional[str] = None
    trading_platforms: List[str] = field(default_factory=list)
    payout_methods: List[str] = field(default_factory=list)
    trading_rules: List[str] = field(default_factory=list)
    promo_codes: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Return the dataclass as a plain dictionary for the database layer."""
        return {
            "name": self.name,
            "country": self.country,
            "founded_year": self.founded_year,
            "tier": self.tier,
            "max_allocation": self.max_allocation,
            "profit_split": self.profit_split,
            "challenge_fee_from": self.challenge_fee_from,
            "payout_frequency": self.payout_frequency,
            "description": self.description,
            "warning_flag": self.warning_flag,
            "warning_message": self.warning_message,
            "trading_platforms": self.trading_platforms,
            "payout_methods": self.payout_methods,
            "trading_rules": self.trading_rules,
            "promo_codes": self.promo_codes,
        }


class ScraperError(RuntimeError):
    """Raised for unrecoverable scraping failures (network, parsing, etc.)."""


class PropFirmScraper:
    """Scrapes propfirmmatch.com (and firm sites) for prop firm data."""

    def __init__(
        self,
        base_url: str,
        min_interval: float = 2.0,
        request_timeout: float = 15.0,
        respect_robots: bool = True,
    ) -> None:
        """Create the scraper.

        Args:
            base_url: Base URL of propfirmmatch.com.
            min_interval: Minimum seconds between outbound requests.
            request_timeout: Per-request timeout in seconds.
            respect_robots: Whether to honour the site's robots.txt.
        """
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = request_timeout
        self._respect_robots = respect_robots
        self._rate_limiter = RateLimiter(min_interval)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

        self._robots: Optional[RobotFileParser] = None
        if respect_robots:
            self._robots = self._load_robots()

    # -- public API --------------------------------------------------------

    def is_firm_listed(self, firm_name: str) -> bool:
        """Return whether the firm appears to be listed on propfirmmatch.com.

        The check tries a direct firm page (via a slugified URL) and falls back
        to scanning the firms directory listing for a matching name.

        Args:
            firm_name: The firm name to validate.

        Returns:
            ``True`` if the firm appears to be listed, ``False`` otherwise.
        """
        firm_name = firm_name.strip()
        if not firm_name:
            return False

        # Strategy 1: direct firm page via slug.
        firm_url = self._firm_page_url(firm_name)
        html = self._get(firm_url)
        if html and self._page_matches_firm(html, firm_name):
            logger.debug("Firm '%s' validated via direct page.", firm_name)
            return True

        # Strategy 2: scan the directory listing for the name.
        listed = self._scan_directory_for_firm(firm_name)
        logger.debug("Firm '%s' directory validation result: %s", firm_name, listed)
        return listed

    def scrape_firm(self, firm_name: str) -> Optional[FirmData]:
        """Scrape structured data for a listed firm.

        Args:
            firm_name: The firm to scrape.

        Returns:
            A populated :class:`FirmData` instance, or ``None`` if the firm's
            page could not be retrieved.
        """
        firm_name = firm_name.strip()
        firm_url = self._firm_page_url(firm_name)
        html = self._get(firm_url)
        if not html:
            logger.warning("Could not retrieve page for firm '%s'.", firm_name)
            return None

        try:
            return self._parse_firm_page(html, firm_name)
        except Exception:  # noqa: BLE001 - parsing is inherently brittle.
            logger.exception("Failed to parse firm page for '%s'.", firm_name)
            # Return a minimal record so the firm still lands in the DB.
            return FirmData(name=firm_name)

    # -- HTTP helpers ------------------------------------------------------

    def _load_robots(self) -> Optional[RobotFileParser]:
        """Fetch and parse robots.txt for the base host, if available."""
        robots_url = urljoin(self._base_url, "/robots.txt")
        parser = RobotFileParser()
        try:
            self._rate_limiter.wait()
            resp = self._session.get(robots_url, timeout=self._timeout)
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
                logger.info("Loaded robots.txt from %s", robots_url)
                return parser
            logger.info(
                "No robots.txt available (HTTP %s); proceeding politely.",
                resp.status_code,
            )
        except requests.RequestException as exc:
            logger.warning("Could not load robots.txt: %s", exc)
        return None

    def _allowed(self, url: str) -> bool:
        """Return whether fetching ``url`` is permitted by robots.txt."""
        if not self._respect_robots or self._robots is None:
            return True
        try:
            return self._robots.can_fetch(USER_AGENT, url)
        except Exception:  # noqa: BLE001 - be permissive if parser errors.
            return True

    def _get(self, url: str, retries: int = 3) -> Optional[str]:
        """Fetch a URL with rate limiting, retries and backoff.

        Args:
            url: The absolute URL to fetch.
            retries: Number of attempts before giving up.

        Returns:
            The response text on success, or ``None`` on failure / disallowed.
        """
        if not self._allowed(url):
            logger.info("robots.txt disallows fetching %s; skipping.", url)
            return None

        backoff = 2.0
        for attempt in range(1, retries + 1):
            try:
                self._rate_limiter.wait()
                resp = self._session.get(url, timeout=self._timeout)
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code == 404:
                    logger.debug("404 Not Found for %s", url)
                    return None
                if resp.status_code == 429:
                    # Respect an explicit rate-limit signal.
                    retry_after = float(
                        resp.headers.get("Retry-After", backoff)
                    )
                    logger.warning(
                        "Rate limited (429) on %s; sleeping %.1fs.",
                        url,
                        retry_after,
                    )
                    time.sleep(retry_after)
                else:
                    logger.warning(
                        "Unexpected HTTP %s for %s (attempt %s/%s).",
                        resp.status_code,
                        url,
                        attempt,
                        retries,
                    )
                    time.sleep(backoff)
            except requests.RequestException as exc:
                logger.warning(
                    "Request error for %s (attempt %s/%s): %s",
                    url,
                    attempt,
                    retries,
                    exc,
                )
                time.sleep(backoff)
            backoff *= 2
        logger.error("Giving up on %s after %s attempts.", url, retries)
        return None

    # -- URL / matching helpers -------------------------------------------

    @staticmethod
    def slugify(firm_name: str) -> str:
        """Convert a firm name into a URL-friendly slug.

        Args:
            firm_name: The human-readable firm name.

        Returns:
            A lowercase, hyphen-separated slug.
        """
        slug = firm_name.strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        return slug.strip("-")

    def _firm_page_url(self, firm_name: str) -> str:
        """Build the candidate firm page URL for a name."""
        return urljoin(self._base_url, f"prop-firm/{self.slugify(firm_name)}")

    @staticmethod
    def _normalise(text: str) -> str:
        """Normalise text for loose comparison (lowercase alphanumerics)."""
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def _page_matches_firm(self, html: str, firm_name: str) -> bool:
        """Heuristically confirm a fetched page really is about the firm."""
        soup = BeautifulSoup(html, "html.parser")
        target = self._normalise(firm_name)
        if not target:
            return False

        # Check the <title> and primary heading for the firm name.
        candidates: List[str] = []
        if soup.title and soup.title.string:
            candidates.append(soup.title.string)
        heading = soup.find(["h1", "h2"])
        if heading:
            candidates.append(heading.get_text(" ", strip=True))

        return any(target in self._normalise(c) for c in candidates)

    def _scan_directory_for_firm(self, firm_name: str) -> bool:
        """Scan the firms directory pages for a matching firm name."""
        target = self._normalise(firm_name)
        for path in ("prop-firms", "prop-firm", "firms", ""):
            url = urljoin(self._base_url, path)
            html = self._get(url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a"):
                text = link.get_text(" ", strip=True)
                href = link.get("href", "")
                if target and (
                    target in self._normalise(text)
                    or target in self._normalise(href)
                ):
                    return True
        return False

    # -- parsing -----------------------------------------------------------

    def _parse_firm_page(self, html: str, firm_name: str) -> FirmData:
        """Extract a :class:`FirmData` from a firm's propfirmmatch.com page.

        The extraction is heuristic and resilient: it pulls values from labelled
        key/value pairs, tables and free text, and silently omits anything it
        cannot confidently determine.
        """
        soup = BeautifulSoup(html, "html.parser")
        data = FirmData(name=firm_name)

        # Prefer the on-page heading as the canonical display name.
        heading = soup.find(["h1", "h2"])
        if heading:
            heading_text = heading.get_text(" ", strip=True)
            if heading_text and self._normalise(firm_name) in self._normalise(
                heading_text
            ):
                data.name = heading_text

        pairs = self._extract_key_value_pairs(soup)

        data.country = self._first_match(pairs, ["country", "location", "based"])
        data.tier = self._first_match(pairs, ["tier", "rating", "grade"])
        data.payout_frequency = self._first_match(
            pairs, ["payout frequency", "payout", "withdrawal frequency"]
        )

        founded = self._first_match(pairs, ["founded", "established", "year"])
        data.founded_year = self._parse_year(founded)

        allocation = self._first_match(
            pairs, ["max allocation", "maximum allocation", "capital", "funding"]
        )
        data.max_allocation = self._parse_money_int(allocation)

        split = self._first_match(pairs, ["profit split", "split"])
        data.profit_split = self._parse_percent(split)

        fee = self._first_match(
            pairs, ["challenge fee", "fee from", "starting fee", "price from"]
        )
        data.challenge_fee_from = self._parse_money_float(fee)

        data.trading_platforms = self._split_list(
            self._first_match(pairs, ["platform", "platforms", "trading platform"])
        )
        data.payout_methods = self._split_list(
            self._first_match(pairs, ["payout method", "withdrawal method", "payout methods"])
        )

        # Description: first substantial paragraph on the page.
        for paragraph in soup.find_all("p"):
            text = paragraph.get_text(" ", strip=True)
            if len(text) >= 60:
                data.description = text[:2000]
                break

        # Warnings: look for scam / caution / avoid language.
        warning = self._detect_warning(soup)
        if warning:
            data.warning_flag = True
            data.warning_message = warning

        return data

    @staticmethod
    def _extract_key_value_pairs(soup: BeautifulSoup) -> Dict[str, str]:
        """Extract labelled key/value pairs from tables and definition lists."""
        pairs: Dict[str, str] = {}

        # HTML tables: two-column rows are treated as label -> value.
        for row in soup.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2:
                key = cells[0].get_text(" ", strip=True).lower()
                value = cells[1].get_text(" ", strip=True)
                if key and value:
                    pairs.setdefault(key, value)

        # Definition lists.
        for dl in soup.find_all("dl"):
            terms = dl.find_all("dt")
            defs = dl.find_all("dd")
            for term, definition in zip(terms, defs):
                key = term.get_text(" ", strip=True).lower()
                value = definition.get_text(" ", strip=True)
                if key and value:
                    pairs.setdefault(key, value)

        return pairs

    @staticmethod
    def _first_match(pairs: Dict[str, str], keywords: List[str]) -> Optional[str]:
        """Return the first pair value whose key contains any keyword."""
        for keyword in keywords:
            for key, value in pairs.items():
                if keyword in key:
                    return value
        return None

    @staticmethod
    def _parse_year(text: Optional[str]) -> Optional[int]:
        """Parse a four-digit year from free text."""
        if not text:
            return None
        match = re.search(r"(19|20)\d{2}", text)
        return int(match.group(0)) if match else None

    @staticmethod
    def _parse_percent(text: Optional[str]) -> Optional[int]:
        """Parse a percentage value from free text."""
        if not text:
            return None
        match = re.search(r"(\d{1,3})\s*%", text)
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_money_int(text: Optional[str]) -> Optional[int]:
        """Parse a monetary amount into an integer, honouring k/m suffixes."""
        value = PropFirmScraper._parse_money_float(text)
        return int(value) if value is not None else None

    @staticmethod
    def _parse_money_float(text: Optional[str]) -> Optional[float]:
        """Parse a monetary amount into a float, honouring k/m suffixes."""
        if not text:
            return None
        match = re.search(
            r"\$?\s*([\d,]+(?:\.\d+)?)\s*([kmKM])?", text.replace(",", "")
        )
        if not match:
            return None
        try:
            amount = float(match.group(1))
        except ValueError:
            return None
        suffix = (match.group(2) or "").lower()
        if suffix == "k":
            amount *= 1_000
        elif suffix == "m":
            amount *= 1_000_000
        return amount

    @staticmethod
    def _split_list(text: Optional[str]) -> List[str]:
        """Split a delimited string into a clean list of values."""
        if not text:
            return []
        parts = re.split(r"[,/|;]| and ", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _detect_warning(soup: BeautifulSoup) -> Optional[str]:
        """Detect scam/caution warnings in the page text."""
        warning_terms = (
            "scam",
            "do not touch",
            "avoid",
            "warning",
            "caution",
            "not paying",
            "banned",
            "shut down",
            "closed down",
        )
        for element in soup.find_all(["p", "span", "div", "li"]):
            text = element.get_text(" ", strip=True)
            lowered = text.lower()
            if 0 < len(text) <= 300 and any(term in lowered for term in warning_terms):
                return text
        return None

    def check_connection(self) -> bool:
        """Verify connectivity to the base site (used during startup checks)."""
        html = self._get(self._base_url)
        return html is not None

    def close(self) -> None:
        """Close the underlying requests session."""
        self._session.close()
