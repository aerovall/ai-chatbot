"""Seed data for the TickShift knowledge base.

Populates the database with the curated set of prop firms and their active promo
codes from the TickShift catalogue (https://www.tickshift.app/). This is the
authoritative baseline the bot serves for commands like ``!promo`` and ``!firm``.

The seed is *idempotent*: it upserts by firm name, so running it repeatedly (for
example on every startup) refreshes the curated values without creating
duplicates. Firms discovered dynamically at runtime are left untouched.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from database import Database

logger = logging.getLogger("tickshift.seed")


# Curated firm catalogue from https://www.tickshift.app/.
# Only firms with an active promo carry a ``promo_codes`` entry; every firm still
# gets its headline attributes so !firm and !ask have real data to work with.
SEED_FIRMS: List[Dict[str, object]] = [
    {
        "name": "Lucid Trading",
        "country": "United States",
        "founded_year": 2024,
        "tier": "S",
        "max_allocation": 600_000,
        "promo_codes": [
            {"code": "TICKSHIFT", "discount_percentage": 40, "is_active": True}
        ],
    },
    {
        "name": "Tradeify",
        "country": "United States",
        "founded_year": 2023,
        "tier": "S",
        "max_allocation": 750_000,
    },
    {
        "name": "My Funded Futures",
        "country": "Canada",
        "founded_year": 2023,
        "tier": "S",
        "max_allocation": 600_000,
        "promo_codes": [
            {"code": "TICKSHIFT", "discount_percentage": 20, "is_active": True}
        ],
    },
    {
        "name": "Funding Predicts",
        "country": "United States",
        "founded_year": 2026,
        "tier": "S",
        "max_allocation": 150_000,
        "promo_codes": [
            {"code": "TICKSHIFT", "discount_percentage": 30, "is_active": True}
        ],
    },
    {
        "name": "Apex Trader Funding",
        "country": "United States",
        "founded_year": 2021,
        "tier": "A",
        "max_allocation": 3_000_000,
    },
    {
        "name": "Topstep",
        "country": "United States",
        "founded_year": 2012,
        "tier": "A",
        "max_allocation": 450_000,
    },
    {
        "name": "Take Profit Trader",
        "country": "United States",
        "founded_year": 2021,
        "tier": "B",
        "max_allocation": 450_000,
    },
    {
        "name": "FundedNext",
        "country": "United Arab Emirates",
        "founded_year": 2022,
        "tier": "B",
        "max_allocation": 300_000,
        "promo_codes": [
            {"code": "REFICOFAR", "discount_percentage": 30, "is_active": True}
        ],
    },
    {
        "name": "FXIFY",
        "country": "United Kingdom",
        "founded_year": 2023,
        "tier": "B",
        "max_allocation": 400_000,
        "promo_codes": [
            {"code": "TICKSHIFT", "discount_percentage": 30, "is_active": True}
        ],
    },
    {
        "name": "TradeDay",
        "country": "United Kingdom",
        "founded_year": 2020,
        "tier": "C",
        "max_allocation": 400_000,
    },
    {
        "name": "Alpha Futures",
        "country": "Netherlands",
        "founded_year": 2024,
        "tier": "D",
        "max_allocation": 450_000,
        "warning_flag": True,
        "warning_message": "Do Not Touch — flagged as problematic in the TickShift catalogue.",
        "promo_codes": [
            {"code": "170043", "discount_percentage": 25, "is_active": True}
        ],
    },
]


def seed_database(db: Database, firms: List[Dict[str, object]] = None) -> int:
    """Upsert the curated firm catalogue into the database.

    Args:
        db: The database helper.
        firms: Optional override list of firm dictionaries. Defaults to
            :data:`SEED_FIRMS`.

    Returns:
        The number of firms successfully seeded.
    """
    firms = firms if firms is not None else SEED_FIRMS
    seeded = 0
    for firm in firms:
        try:
            db.upsert_firm(firm)
            seeded += 1
        except Exception:  # noqa: BLE001 - one bad row shouldn't stop seeding.
            logger.exception("Failed to seed firm '%s'.", firm.get("name"))
    logger.info("Seeded %d/%d firms from the TickShift catalogue.", seeded, len(firms))
    return seeded


def main() -> None:
    """Run seeding as a standalone script using the app configuration."""
    from config import Config, configure_logging

    config = Config.load()
    configure_logging(config)
    db = Database(config.database_url, refresh_days=config.search_refresh_days)
    db.init()
    seed_database(db)


if __name__ == "__main__":
    main()
