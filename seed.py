"""Seed data for the TickShift knowledge base.

Populates the database with the full prop firm catalogue from the TickShift
website (https://www.tickshift.app/). The authoritative data is bundled as
``data/tickshift_data.json`` (extracted from the TickShift website's dataset)
and transformed here into the bot's database schema.

The seed is *idempotent*: it upserts by firm name, so running it repeatedly (for
example on every startup) refreshes the curated values without creating
duplicates. Firms discovered dynamically at runtime are left untouched.

To refresh the catalogue, replace ``data/tickshift_data.json`` with a newer
export and redeploy — no code changes are required.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional

from database import Database

logger = logging.getLogger("tickshift.seed")

# Path to the bundled catalogue, resolved relative to this module.
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tickshift_data.json")

# Human-readable labels for the trading-rule fields in the source data.
_RULE_LABELS = {
    "weekendHolding": "Weekend holding",
    "overnightHolding": "Overnight holding",
    "hedging": "Hedging",
    "copyTrading": "Copy trading",
    "eas": "EAs/bots",
}

# Platform keys that indicate forex/CFD (as opposed to futures) trading.
_FOREX_PLATFORM_KEYS = {"mt4", "mt5", "dxtrade"}


def _firm_market(slug: str) -> str:
    """Classify a firm's market, mirroring the TickShift website's logic.

    Args:
        slug: The firm slug.

    Returns:
        ``"futures"``, ``"cfd"`` or ``"predictions"``.
    """
    if slug == "funding-predicts":
        return "predictions"
    if slug == "fxify":
        return "cfd"
    return "futures"


def _account_market(slug: str, account_type_name: str) -> str:
    """Classify a single account type's market (website ``marketOf`` logic)."""
    if slug == "funding-predicts":
        return "predictions"
    if slug == "fxify" or "cfd" in (account_type_name or "").lower():
        return "cfd"
    return "futures"


def _parse_allocation(value: Optional[str]) -> Optional[int]:
    """Parse a max-allocation string like ``"$600K"`` or ``"$3M"`` into an int."""
    if not value:
        return None
    match = re.search(r"([\d.]+)\s*([KMkm]?)", value.replace(",", ""))
    if not match:
        return None
    amount = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        amount *= 1_000
    elif suffix == "m":
        amount *= 1_000_000
    return int(amount)


def _transform_firm(
    firm: Dict[str, object],
    tiers: Dict[str, str],
    warnings: Dict[str, str],
    platform_labels: Dict[str, str],
    payout_labels: Dict[str, str],
) -> Dict[str, object]:
    """Transform one website firm record into the bot's firm dictionary.

    Args:
        firm: A firm record from the bundled TickShift dataset.
        tiers: Mapping of firm slug to tier letter.
        warnings: Mapping of firm slug to a public warning source URL.
        platform_labels: Mapping of platform key to display label.
        payout_labels: Mapping of payout-method key to display label.

    Returns:
        A firm dictionary compatible with :meth:`Database.upsert_firm`.
    """
    slug = str(firm.get("slug", ""))
    market = _firm_market(slug)
    account_types = firm.get("accountTypes", []) or []
    platform_keys = firm.get("platforms", []) or []

    # For futures firms, keep only the futures side: drop CFD/forex account types
    # and forex/CFD platforms. Non-futures firms keep their full data (they are
    # hidden by the futures-only read filter, but stay intact if it is disabled).
    if market == "futures":
        account_types = [
            at
            for at in account_types
            if _account_market(slug, at.get("name", "")) == "futures"
        ]
        platform_keys = [
            p for p in platform_keys if p not in _FOREX_PLATFORM_KEYS
        ]

    # Flatten all account sizes to derive headline profit split / cheapest fee.
    sizes = [s for at in account_types for s in at.get("sizes", [])]
    prices = [s["price"] for s in sizes if s.get("price") is not None]
    splits = [s["profitSplit"] for s in sizes if s.get("profitSplit") is not None]

    # Build trading-rule detail lines from the rule flags and account summaries.
    rules: List[str] = []
    trading_rules = firm.get("tradingRules", {}) or {}
    for key, label in _RULE_LABELS.items():
        if trading_rules.get(key):
            rules.append(f"{label}: {trading_rules[key]}")
    for at in account_types:
        at_sizes = at.get("sizes", [])
        labels = [s.get("size") for s in at_sizes if s.get("size")]
        at_prices = [s["price"] for s in at_sizes if s.get("price") is not None]
        price_range = (
            f"${min(at_prices):g}–${max(at_prices):g}" if at_prices else "n/a"
        )
        rules.append(
            f"Account '{at.get('name')}': {at.get('drawdownMode')} drawdown, "
            f"news {at.get('newsTrading')}, consistency {at.get('consistencyRule')}, "
            f"sizes {'/'.join(labels)} ({price_range})"
        )

    result: Dict[str, object] = {
        "name": firm.get("name"),
        "country": firm.get("country"),
        "founded_year": firm.get("founded"),
        "tier": tiers.get(slug),
        "market": market,
        "max_allocation": _parse_allocation(firm.get("maxAllocation")),
        "profit_split": max(splits) if splits else None,
        "challenge_fee_from": min(prices) if prices else None,
        "payout_frequency": firm.get("payoutFrequency"),
        "description": firm.get("description") or firm.get("tagline"),
        "trading_platforms": [
            platform_labels.get(p, p) for p in platform_keys
        ],
        "payout_methods": [
            payout_labels.get(p, p) for p in firm.get("payoutMethods", []) or []
        ],
        "trading_rules": rules,
    }

    if slug in warnings:
        # A public warning exists for this firm. Deliberately omit the source
        # link: it points at an industry watchdog we must never surface to
        # users, so we keep the message generic.
        result["warning_flag"] = True
        result["warning_message"] = (
            "This firm has an active public warning in the industry — "
            "exercise caution and do your own due diligence."
        )

    if firm.get("promoCode"):
        result["promo_codes"] = [
            {
                "code": firm["promoCode"],
                "discount_percentage": firm.get("discountPct"),
                "is_active": True,
            }
        ]

    return result


def load_seed_firms(data_path: str = DATA_PATH) -> List[Dict[str, object]]:
    """Load and transform the bundled TickShift catalogue.

    Args:
        data_path: Path to the bundled dataset JSON.

    Returns:
        A list of firm dictionaries ready for :meth:`Database.upsert_firm`.
        Returns an empty list if the file is missing or malformed.
    """
    try:
        with open(data_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        logger.exception("Could not load seed data from %s.", data_path)
        return []

    tiers = data.get("tiers", {}) or {}
    warnings = data.get("warnings", {}) or {}
    platform_labels = data.get("platformLabels", {}) or {}
    payout_labels = data.get("payoutLabels", {}) or {}

    firms: List[Dict[str, object]] = []
    for firm in data.get("firms", []) or []:
        try:
            firms.append(
                _transform_firm(
                    firm, tiers, warnings, platform_labels, payout_labels
                )
            )
        except Exception:  # noqa: BLE001 - skip a malformed record, keep the rest.
            logger.exception(
                "Failed to transform firm '%s'.", firm.get("slug")
            )
    return firms


def seed_database(
    db: Database, firms: Optional[List[Dict[str, object]]] = None
) -> int:
    """Upsert the TickShift firm catalogue into the database.

    Args:
        db: The database helper.
        firms: Optional override list of firm dictionaries. Defaults to the
            bundled catalogue loaded via :func:`load_seed_firms`.

    Returns:
        The number of firms successfully seeded.
    """
    firms = firms if firms is not None else load_seed_firms()
    if not firms:
        logger.warning("No seed firms available; skipping catalogue seed.")
        return 0

    seeded = 0
    for firm in firms:
        try:
            db.upsert_firm(firm)
            seeded += 1
        except Exception:  # noqa: BLE001 - one bad row shouldn't stop seeding.
            logger.exception("Failed to seed firm '%s'.", firm.get("name"))
    logger.info(
        "Seeded %d/%d firms from the TickShift catalogue.", seeded, len(firms)
    )
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
