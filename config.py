"""Centralised configuration management for the TickShift Discord bot.

All configuration is loaded from environment variables (optionally via a local
``.env`` file). Required variables are validated eagerly so the bot fails fast
with a clear, actionable error message instead of crashing deep in the runtime.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load variables from a local .env file if present. Real environment variables
# always take precedence over values defined in the file.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when the environment is missing required configuration."""


def _get_required(name: str) -> str:
    """Return a required environment variable or record it as missing.

    Args:
        name: The environment variable name.

    Returns:
        The variable's value, or an empty string if it is unset. Missing
        variables are collected and validated together in :meth:`Config.load`.
    """
    return os.getenv(name, "").strip()


def _get_float(name: str, default: float) -> float:
    """Parse a float environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    """Parse an int environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable, falling back to ``default``.

    Truthy values (case-insensitive): ``1``, ``true``, ``yes``, ``on``.
    Falsy values: ``0``, ``false``, ``no``, ``off``.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_name_set(name: str, default: str = "") -> frozenset:
    """Parse a comma-separated list of names into a lowercased frozenset.

    Args:
        name: The environment variable name.
        default: Default comma-separated value if the variable is unset.
    """
    raw = os.getenv(name)
    if raw is None:
        raw = default
    names = {
        token.strip().lower()
        for token in raw.replace(";", ",").split(",")
        if token.strip()
    }
    return frozenset(names)


def _get_id_set(name: str) -> frozenset:
    """Parse a comma-separated list of integer IDs into a frozenset.

    Non-numeric entries are ignored. An empty/unset value yields an empty set
    (interpreted downstream as "no restriction").
    """
    raw = os.getenv(name, "")
    ids = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit():
            ids.add(int(token))
    return frozenset(ids)


@dataclass(frozen=True)
class Config:
    """Immutable, validated view of the bot's runtime configuration."""

    # Required secrets / connection strings.
    discord_token: str
    anthropic_api_key: str
    database_url: str
    propfirmmatch_base_url: str

    # Optional, tunable values with sensible defaults.
    anthropic_model: str = "claude-sonnet-5"
    anthropic_max_tokens: int = 1024
    command_prefix: str = "!"
    scrape_min_interval: float = 2.0
    search_refresh_days: int = 30
    seed_on_startup: bool = True
    qa_cache_enabled: bool = True
    qa_cache_ttl_days: int = 0
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.85
    voyage_api_key: str = ""
    voyage_model: str = "voyage-3.5"
    # Security / abuse guardrails.
    max_question_length: int = 500
    rate_limit_per_user: int = 5
    rate_limit_window: int = 60
    rate_limit_global: int = 30
    allowed_channel_ids: frozenset = frozenset()
    ignore_dms: bool = False
    futures_only: bool = True
    included_firm_names: frozenset = frozenset()
    log_level: str = "INFO"
    log_file: str = "logs/bot.log"

    # Populated during validation; not part of the public constructor surface.
    _missing: List[str] = field(default_factory=list, repr=False, compare=False)

    @classmethod
    def load(cls) -> "Config":
        """Build a :class:`Config` from the environment and validate it.

        Returns:
            A fully validated configuration instance.

        Raises:
            ConfigError: If one or more required environment variables are
                missing or empty.
        """
        discord_token = _get_required("DISCORD_TOKEN")
        anthropic_api_key = _get_required("ANTHROPIC_API_KEY")
        database_url = _get_required("DATABASE_URL")
        propfirmmatch_base_url = (
            _get_required("PROPFIRMMATCH_BASE_URL") or "https://propfirmmatch.com/"
        )

        missing: List[str] = []
        if not discord_token:
            missing.append("DISCORD_TOKEN")
        if not anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not database_url:
            missing.append("DATABASE_URL")

        if missing:
            raise ConfigError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill in the values."
            )

        return cls(
            discord_token=discord_token,
            anthropic_api_key=anthropic_api_key,
            database_url=database_url,
            propfirmmatch_base_url=propfirmmatch_base_url.rstrip("/") + "/",
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5").strip()
            or "claude-sonnet-5",
            anthropic_max_tokens=_get_int("ANTHROPIC_MAX_TOKENS", 1024),
            command_prefix=os.getenv("COMMAND_PREFIX", "!").strip() or "!",
            scrape_min_interval=_get_float("SCRAPE_MIN_INTERVAL", 2.0),
            search_refresh_days=_get_int("SEARCH_REFRESH_DAYS", 30),
            seed_on_startup=_get_bool("SEED_ON_STARTUP", True),
            qa_cache_enabled=_get_bool("QA_CACHE_ENABLED", True),
            qa_cache_ttl_days=_get_int("QA_CACHE_TTL_DAYS", 0),
            semantic_cache_enabled=_get_bool("SEMANTIC_CACHE_ENABLED", True),
            semantic_cache_threshold=_get_float("SEMANTIC_CACHE_THRESHOLD", 0.85),
            voyage_api_key=_get_required("VOYAGE_API_KEY"),
            voyage_model=os.getenv("VOYAGE_MODEL", "voyage-3.5").strip()
            or "voyage-3.5",
            max_question_length=_get_int("MAX_QUESTION_LENGTH", 500),
            rate_limit_per_user=_get_int("RATE_LIMIT_PER_USER", 5),
            rate_limit_window=_get_int("RATE_LIMIT_WINDOW", 60),
            rate_limit_global=_get_int("RATE_LIMIT_GLOBAL", 30),
            allowed_channel_ids=_get_id_set("ALLOWED_CHANNEL_IDS"),
            ignore_dms=_get_bool("IGNORE_DMS", False),
            futures_only=_get_bool("FUTURES_ONLY", True),
            included_firm_names=_get_name_set("INCLUDED_FIRMS", "FXIFY"),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
            log_file=os.getenv("LOG_FILE", "logs/bot.log").strip() or "logs/bot.log",
        )


def configure_logging(config: Config) -> logging.Logger:
    """Configure application-wide logging to both stdout and a rotating file.

    Args:
        config: The loaded configuration providing log level and file path.

    Returns:
        The configured root ``tickshift`` logger.
    """
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, config.log_level, logging.INFO)

    logger = logging.getLogger("tickshift")
    logger.setLevel(level)
    logger.propagate = False

    # Avoid installing duplicate handlers if this is called more than once.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
