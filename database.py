"""Database layer for the TickShift Discord bot.

Defines the SQLAlchemy ORM models that mirror the PostgreSQL schema and a
:class:`Database` helper that owns the engine/session factory and exposes the
high-level query functions used by the rest of the application.

The models intentionally mirror the SQL DDL from the project specification. The
``Database.init`` method issues ``CREATE TABLE IF NOT EXISTS`` for every model so
a fresh PostgreSQL database is bootstrapped automatically on first run.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

logger = logging.getLogger("tickshift.database")


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


class PropFirm(Base):
    """A prop trading firm and its headline attributes."""

    __tablename__ = "prop_firms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    country: Mapped[Optional[str]] = mapped_column(String(100))
    founded_year: Mapped[Optional[int]] = mapped_column(Integer)
    tier: Mapped[Optional[str]] = mapped_column(String(10))
    max_allocation: Mapped[Optional[int]] = mapped_column(Integer)
    profit_split: Mapped[Optional[int]] = mapped_column(Integer)
    challenge_fee_from: Mapped[Optional[float]] = mapped_column(Numeric(10, 2))
    payout_frequency: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    warning_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    warning_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    details: Mapped[List["FirmDetail"]] = relationship(
        back_populates="firm", cascade="all, delete-orphan"
    )
    promo_codes: Mapped[List["PromoCode"]] = relationship(
        back_populates="firm", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, object]:
        """Serialise the firm (and its related rows) into a plain dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "country": self.country,
            "founded_year": self.founded_year,
            "tier": self.tier,
            "max_allocation": self.max_allocation,
            "profit_split": self.profit_split,
            "challenge_fee_from": (
                float(self.challenge_fee_from)
                if self.challenge_fee_from is not None
                else None
            ),
            "payout_frequency": self.payout_frequency,
            "description": self.description,
            "warning_flag": self.warning_flag,
            "warning_message": self.warning_message,
            "details": [d.to_dict() for d in self.details],
            "promo_codes": [
                p.to_dict() for p in self.promo_codes if p.is_active
            ],
        }


class FirmDetail(Base):
    """A typed key/value attribute attached to a firm.

    Used for flexible attributes that do not warrant a dedicated column, such as
    ``trading_platform``, ``payout_method`` or ``trading_rule``.
    """

    __tablename__ = "firm_details"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    firm_id: Mapped[int] = mapped_column(
        ForeignKey("prop_firms.id", ondelete="CASCADE"), nullable=False
    )
    detail_type: Mapped[Optional[str]] = mapped_column(String(100))
    detail_value: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )

    firm: Mapped["PropFirm"] = relationship(back_populates="details")

    def to_dict(self) -> Dict[str, object]:
        """Serialise the detail row into a plain dictionary."""
        return {"type": self.detail_type, "value": self.detail_value}


class PromoCode(Base):
    """A promotional / discount code offered by a firm."""

    __tablename__ = "promo_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    firm_id: Mapped[int] = mapped_column(
        ForeignKey("prop_firms.id", ondelete="CASCADE"), nullable=False
    )
    code: Mapped[str] = mapped_column(String(100), nullable=False)
    discount_percentage: Mapped[Optional[int]] = mapped_column(Integer)
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    firm: Mapped["PropFirm"] = relationship(back_populates="promo_codes")

    def to_dict(self) -> Dict[str, object]:
        """Serialise the promo code into a plain dictionary."""
        return {
            "code": self.code,
            "discount_percentage": self.discount_percentage,
            "description": self.description,
            "is_active": self.is_active,
        }


class FirmSearchCache(Base):
    """Tracks which firm names have been searched/validated and when.

    This avoids repeatedly scraping propfirmmatch.com for the same firm and
    provides the basis for a periodic refresh mechanism.
    """

    __tablename__ = "firm_search_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    firm_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    is_valid_firm: Mapped[Optional[bool]] = mapped_column(Boolean)
    last_searched: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )
    search_status: Mapped[Optional[str]] = mapped_column(String(50))


class QACache(Base):
    """Caches Claude answers so repeat questions don't re-hit the API.

    Each entry is keyed by a hash of the normalised question and stores the
    fingerprint (``kb_version``) of the knowledge base that produced the answer.
    A cached answer is only reused while that fingerprint still matches the
    current knowledge base, so answers are automatically regenerated whenever the
    underlying firm/promo data changes.
    """

    __tablename__ = "qa_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    question_text: Mapped[Optional[str]] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    kb_version: Mapped[str] = mapped_column(String(64), nullable=False)
    # JSON-encoded embedding vector of the question (for semantic matching).
    embedding: Mapped[Optional[str]] = mapped_column(Text)
    # Command namespace ("ask"/"compare") so different commands never cross-match.
    namespace: Mapped[Optional[str]] = mapped_column(
        String(20), default="ask", index=True
    )
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )


class Database:
    """Owns the SQLAlchemy engine and exposes high-level query helpers."""

    def __init__(self, database_url: str, refresh_days: int = 30) -> None:
        """Create the engine and session factory.

        Args:
            database_url: A SQLAlchemy-compatible PostgreSQL connection string.
            refresh_days: Number of days after which a cached search entry is
                considered stale.
        """
        self._refresh_days = refresh_days
        # pool_pre_ping avoids handing out dead connections after DB restarts.
        self._engine = create_engine(
            database_url, pool_pre_ping=True, future=True
        )
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False, future=True
        )

    # -- lifecycle ---------------------------------------------------------

    def init(self) -> None:
        """Create all tables if they do not already exist and run migrations."""
        Base.metadata.create_all(self._engine)
        self._migrate()
        logger.info("Database schema verified / created.")

    def _migrate(self) -> None:
        """Apply lightweight, idempotent schema migrations.

        ``create_all`` does not add new columns to tables that already exist, so
        for PostgreSQL we additively ensure the semantic-cache columns are
        present. This is safe to run on every startup.
        """
        if self._engine.dialect.name != "postgresql":
            return  # Fresh SQLite/other DBs already have the current columns.
        statements = (
            "ALTER TABLE qa_cache ADD COLUMN IF NOT EXISTS embedding TEXT",
            "ALTER TABLE qa_cache ADD COLUMN IF NOT EXISTS namespace VARCHAR(20)",
        )
        try:
            with self._engine.begin() as conn:
                for statement in statements:
                    conn.execute(text(statement))
            logger.debug("qa_cache semantic-cache columns verified.")
        except Exception:  # noqa: BLE001 - migration must not block startup.
            logger.exception("qa_cache column migration failed (continuing).")

    def verify_connection(self) -> None:
        """Open a connection to confirm the database is reachable.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the connection cannot be made.
        """
        with self._engine.connect() as conn:
            conn.execute(select(1))
        logger.info("Database connection verified.")

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a transactional scope around a series of operations."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -- firm queries ------------------------------------------------------

    def get_all_firms(self) -> List[Dict[str, object]]:
        """Return every firm (with details and active promos) as dictionaries."""
        with self.session() as session:
            firms = session.scalars(select(PropFirm)).all()
            return [firm.to_dict() for firm in firms]

    def get_firm_by_name(self, name: str) -> Optional[Dict[str, object]]:
        """Return a single firm by case-insensitive name, or ``None``.

        Args:
            name: The firm name to look up.
        """
        with self.session() as session:
            firm = self._find_firm(session, name)
            return firm.to_dict() if firm else None

    def firm_exists(self, name: str) -> bool:
        """Return whether a firm with the given name exists locally."""
        with self.session() as session:
            return self._find_firm(session, name) is not None

    @staticmethod
    def _find_firm(session: Session, name: str) -> Optional[PropFirm]:
        """Case-insensitive firm lookup helper bound to an open session."""
        return session.scalars(
            select(PropFirm).where(func.lower(PropFirm.name) == name.strip().lower())
        ).first()

    def upsert_firm(self, data: Dict[str, object]) -> int:
        """Insert or update a firm and its related detail/promo rows.

        Args:
            data: A dictionary describing the firm. Recognised keys include the
                :class:`PropFirm` columns plus optional ``trading_platforms``,
                ``payout_methods``, ``trading_rules`` (lists of strings) and
                ``promo_codes`` (list of dicts).

        Returns:
            The primary key of the inserted or updated firm.
        """
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError("Cannot upsert a firm without a name.")

        with self.session() as session:
            firm = self._find_firm(session, name)
            if firm is None:
                firm = PropFirm(name=name)
                session.add(firm)

            # Update headline columns when provided (preserve existing on None).
            for column in (
                "country",
                "founded_year",
                "tier",
                "max_allocation",
                "profit_split",
                "challenge_fee_from",
                "payout_frequency",
                "description",
                "warning_flag",
                "warning_message",
            ):
                if column in data and data[column] is not None:
                    setattr(firm, column, data[column])

            session.flush()  # Ensure firm.id is populated for child rows.

            # Replace detail rows for the categories we manage so repeated
            # scrapes do not accumulate duplicates.
            self._replace_details(session, firm, data)
            self._replace_promos(session, firm, data)

            session.flush()
            firm_id = firm.id
            logger.info("Upserted firm '%s' (id=%s).", name, firm_id)
            return firm_id

    @staticmethod
    def _replace_details(
        session: Session, firm: PropFirm, data: Dict[str, object]
    ) -> None:
        """Rebuild the flexible detail rows for a firm from scraped data."""
        managed_types = {
            "trading_platform": data.get("trading_platforms"),
            "payout_method": data.get("payout_methods"),
            "trading_rule": data.get("trading_rules"),
        }
        # Only touch categories that were actually provided this time.
        provided = {k: v for k, v in managed_types.items() if v}
        if not provided:
            return

        for detail in list(firm.details):
            if detail.detail_type in provided:
                session.delete(detail)
        session.flush()

        for detail_type, values in provided.items():
            if isinstance(values, str):
                values = [values]
            for value in values:
                value = str(value).strip()
                if value:
                    firm.details.append(
                        FirmDetail(detail_type=detail_type, detail_value=value[:255])
                    )

    @staticmethod
    def _replace_promos(
        session: Session, firm: PropFirm, data: Dict[str, object]
    ) -> None:
        """Rebuild promo code rows for a firm from scraped data."""
        promos = data.get("promo_codes")
        if not promos:
            return

        for promo in list(firm.promo_codes):
            session.delete(promo)
        session.flush()

        for promo in promos:
            if not isinstance(promo, dict):
                continue
            code = str(promo.get("code", "")).strip()
            if not code:
                continue
            firm.promo_codes.append(
                PromoCode(
                    code=code[:100],
                    discount_percentage=promo.get("discount_percentage"),
                    description=promo.get("description"),
                    is_active=bool(promo.get("is_active", True)),
                )
            )

    # -- promo queries -----------------------------------------------------

    def get_active_promo_codes(self) -> List[Dict[str, object]]:
        """Return all active promo codes joined with their firm name."""
        with self.session() as session:
            rows = session.execute(
                select(PromoCode, PropFirm.name)
                .join(PropFirm, PromoCode.firm_id == PropFirm.id)
                .where(PromoCode.is_active.is_(True))
                .order_by(PropFirm.name)
            ).all()
            result: List[Dict[str, object]] = []
            for promo, firm_name in rows:
                entry = promo.to_dict()
                entry["firm_name"] = firm_name
                result.append(entry)
            return result

    # -- search cache ------------------------------------------------------

    def get_cache_entry(self, firm_name: str) -> Optional[Dict[str, object]]:
        """Return the cache entry for a firm name, or ``None`` if absent."""
        with self.session() as session:
            entry = session.scalars(
                select(FirmSearchCache).where(
                    func.lower(FirmSearchCache.firm_name)
                    == firm_name.strip().lower()
                )
            ).first()
            if entry is None:
                return None
            return {
                "firm_name": entry.firm_name,
                "is_valid_firm": entry.is_valid_firm,
                "last_searched": entry.last_searched,
                "search_status": entry.search_status,
            }

    def is_cache_fresh(self, firm_name: str) -> bool:
        """Return whether a cache entry exists and is within the refresh window."""
        entry = self.get_cache_entry(firm_name)
        if entry is None or entry["last_searched"] is None:
            return False
        last_searched = entry["last_searched"]
        # Normalise naive timestamps (PostgreSQL default) to UTC for comparison.
        if last_searched.tzinfo is None:
            last_searched = last_searched.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._refresh_days)
        return last_searched >= cutoff

    def update_cache(
        self, firm_name: str, is_valid_firm: bool, search_status: str
    ) -> None:
        """Insert or update the search cache entry for a firm.

        Args:
            firm_name: The firm name that was searched.
            is_valid_firm: Whether the firm was found on propfirmmatch.com.
            search_status: A short status string, e.g. ``"found"``,
                ``"not_found"`` or ``"error"``.
        """
        with self.session() as session:
            entry = session.scalars(
                select(FirmSearchCache).where(
                    func.lower(FirmSearchCache.firm_name)
                    == firm_name.strip().lower()
                )
            ).first()
            if entry is None:
                entry = FirmSearchCache(firm_name=firm_name.strip())
                session.add(entry)
            entry.is_valid_firm = is_valid_firm
            entry.search_status = search_status
            entry.last_searched = datetime.now(timezone.utc)
            logger.debug(
                "Cache updated for '%s': valid=%s status=%s",
                firm_name,
                is_valid_firm,
                search_status,
            )

    # -- Q&A answer cache --------------------------------------------------

    def get_cached_answer(
        self, question_hash: str, kb_version: str, ttl_days: int = 0
    ) -> Optional[str]:
        """Return a cached answer if one matches the question and current KB.

        Args:
            question_hash: Hash of the normalised question.
            kb_version: Fingerprint of the current knowledge base. A cached
                answer is only returned when its stored fingerprint matches, so
                stale answers (produced before a data change) are ignored.
            ttl_days: Optional maximum age in days. ``0`` disables time-based
                expiry and relies solely on the knowledge base fingerprint.

        Returns:
            The cached answer text, or ``None`` on a cache miss.
        """
        with self.session() as session:
            entry = session.scalars(
                select(QACache).where(QACache.question_hash == question_hash)
            ).first()
            if entry is None or entry.kb_version != kb_version:
                return None

            if ttl_days and ttl_days > 0:
                created = entry.created_at
                if created is not None:
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
                    if created < cutoff:
                        return None

            entry.hit_count = (entry.hit_count or 0) + 1
            logger.info(
                "Q&A cache hit (hits=%s) for question hash %s.",
                entry.hit_count,
                question_hash[:12],
            )
            return entry.answer

    def store_cached_answer(
        self,
        question_hash: str,
        question_text: str,
        answer: str,
        kb_version: str,
        embedding: Optional[List[float]] = None,
        namespace: str = "ask",
    ) -> None:
        """Insert or refresh a cached answer for a question.

        Args:
            question_hash: Hash of the normalised question.
            question_text: The original question text (for readability/debugging).
            answer: The answer to cache.
            kb_version: Fingerprint of the knowledge base that produced it.
            embedding: Optional question embedding vector for semantic matching.
            namespace: Command namespace (e.g. ``"ask"`` or ``"compare"``).
        """
        with self.session() as session:
            entry = session.scalars(
                select(QACache).where(QACache.question_hash == question_hash)
            ).first()
            if entry is None:
                entry = QACache(question_hash=question_hash, hit_count=0)
                session.add(entry)
            entry.question_text = (question_text or "")[:2000]
            entry.answer = answer
            entry.kb_version = kb_version
            entry.namespace = namespace
            if embedding is not None:
                entry.embedding = json.dumps(embedding)
            entry.updated_at = datetime.now(timezone.utc)
            logger.debug("Cached answer stored for hash %s.", question_hash[:12])

    def get_semantic_candidates(
        self, kb_version: str, namespace: str = "ask"
    ) -> List[Dict[str, object]]:
        """Return cached entries eligible for semantic matching.

        Only entries produced under the current knowledge base fingerprint and
        the given namespace, and that carry an embedding, are returned.

        Args:
            kb_version: Current knowledge base fingerprint.
            namespace: Command namespace to match.

        Returns:
            A list of ``{"id", "answer", "embedding"}`` dictionaries, where
            ``embedding`` is a decoded list of floats.
        """
        with self.session() as session:
            rows = session.scalars(
                select(QACache).where(
                    QACache.kb_version == kb_version,
                    QACache.namespace == namespace,
                    QACache.embedding.is_not(None),
                )
            ).all()
            candidates: List[Dict[str, object]] = []
            for row in rows:
                try:
                    vector = json.loads(row.embedding) if row.embedding else None
                except (ValueError, TypeError):
                    vector = None
                if vector:
                    candidates.append(
                        {"id": row.id, "answer": row.answer, "embedding": vector}
                    )
            return candidates

    def register_cache_hit(self, entry_id: int) -> None:
        """Increment the hit counter for a cached entry by primary key."""
        with self.session() as session:
            entry = session.get(QACache, entry_id)
            if entry is not None:
                entry.hit_count = (entry.hit_count or 0) + 1
