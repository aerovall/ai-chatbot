"""TickShift AI Discord bot entry point.

Wires together the configuration, database, scraper, validator, prompt builder
and the Anthropic Claude client, and exposes the user-facing Discord commands:

    !ask <question>       Ask anything about prop firms.
    !compare <a> <b>      Compare two firms side by side.
    !firm <name>          Show a detailed profile for one firm.
    !promo                List all active promo codes.
    !help                 Show usage help.

Design notes:
- The knowledge base is dynamic: every command rebuilds Claude's system prompt
  from the current database contents.
- Any firm the user names is validated internally before it is discussed. The
  validation source is never revealed to users.
- Blocking work (database access, scraping, HTTP) runs in worker threads via
  ``asyncio.to_thread`` so the Discord event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import anthropic
import discord
from discord.ext import commands

from config import Config, ConfigError, configure_logging
from database import Database
from embeddings import EmbeddingClient, cosine_similarity
from faq import FIRM_ALIASES, FAQKnowledgeBase
from prompts import (
    COMPARE_INSTRUCTIONS,
    build_faq_block,
    build_system_prompt,
    wrap_user_question,
)
from scraper import PropFirmScraper
from seed import seed_database
from validator import FirmValidator

DISCORD_MESSAGE_LIMIT = 2000
EMBED_DESCRIPTION_LIMIT = 4096


class PromptBundle(NamedTuple):
    """A built system prompt plus the fingerprint used by the answer cache.

    Attributes:
        system: System prompt as a list of content blocks. The first (stable)
            block carries a ``cache_control`` marker for Anthropic prompt
            caching; an optional second block holds per-question FAQ excerpts.
        fingerprint: Hash of the stable knowledge (catalogue + FAQ version)
            used to validate/invalidate cached answers.
    """

    system: List[Dict[str, object]]
    fingerprint: str


class TickShiftBot(commands.Bot):
    """A Discord bot that answers prop firm questions using Claude."""

    def __init__(
        self,
        config: Config,
        db: Database,
        scraper: PropFirmScraper,
        validator: FirmValidator,
        logger: logging.Logger,
        embedder: Optional[EmbeddingClient] = None,
    ) -> None:
        """Construct the bot and its collaborators.

        Args:
            config: Validated runtime configuration.
            db: Database helper.
            scraper: Prop firm scraper.
            validator: Firm validator.
            logger: Application logger.
            embedder: Optional Voyage embedding client for semantic caching.
        """
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read command text.
        super().__init__(
            # Accept both an @mention and the text prefix (e.g. "!") so users
            # can run commands either way: "!promo" or "@Bot promo".
            command_prefix=commands.when_mentioned_or(config.command_prefix),
            intents=intents,
            help_command=None,  # We provide a custom !help.
            # Hard guarantee the bot can never ping @everyone/@here/roles, even
            # if a firm description or a model response contains such text.
            allowed_mentions=discord.AllowedMentions.none(),
        )
        # Per-user and global rate limiters to prevent spam / credit abuse.
        self._user_cooldown = commands.CooldownMapping.from_cooldown(
            max(1, config.rate_limit_per_user),
            max(1, config.rate_limit_window),
            commands.BucketType.user,
        )
        self._global_cooldown = commands.CooldownMapping.from_cooldown(
            max(1, config.rate_limit_global),
            max(1, config.rate_limit_window),
            commands.BucketType.default,
        )
        self.config = config
        self.db = db
        self.scraper = scraper
        self.validator = validator
        self.log = logger
        self.embedder = embedder
        # FAQ knowledge base for FAQ-first answering (degrades to empty on error).
        self.faq = FAQKnowledgeBase() if config.faq_enabled else None
        self.claude = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    def semantic_enabled(self) -> bool:
        """Return whether semantic caching is active (client + toggle present)."""
        return self.embedder is not None and self.config.semantic_cache_enabled

    def allowed_markets(self) -> Optional[set]:
        """Return the set of markets the bot may surface, or ``None`` for all.

        In futures-only mode the bot serves and discusses only futures firms;
        forex/CFD and prediction-market firms stay in the database but are
        hidden from every user-facing view (except any explicitly included firm).
        """
        return {"futures"} if self.config.futures_only else None

    def included_firms(self) -> Optional[frozenset]:
        """Return firm names shown even when outside the allowed markets.

        These are non-futures firms kept in the lineup with their forex/CFD
        specifics stripped (e.g. FXIFY). Returns ``None`` when not in
        futures-only mode (every firm is already visible).
        """
        if not self.config.futures_only:
            return None
        return self.config.included_firm_names or None

    # -- guardrails --------------------------------------------------------

    def check_rate_limit(self, message: discord.Message) -> Optional[float]:
        """Check per-user and global rate limits without over-consuming tokens.

        Args:
            message: The triggering message (identifies the user/bucket).

        Returns:
            The number of seconds to wait if rate limited, otherwise ``None``.
        """
        user_bucket = self._user_cooldown.get_bucket(message)
        global_bucket = self._global_cooldown.get_bucket(message)
        # Peek first so we don't consume a token when we're going to reject.
        retry = user_bucket.get_retry_after() or global_bucket.get_retry_after()
        if retry:
            return retry
        user_bucket.update_rate_limit()
        global_bucket.update_rate_limit()
        return None

    def is_allowed_context(self, message: discord.Message) -> bool:
        """Return whether the bot should respond in this channel / DM.

        Honours the optional channel allowlist and the DM policy.

        Args:
            message: The incoming message.
        """
        if message.guild is None:
            return not self.config.ignore_dms
        allowed = self.config.allowed_channel_ids
        if allowed and message.channel.id not in allowed:
            return False
        return True

    def semantic_lookup(
        self, kb_version: str, query_embedding: List[float]
    ) -> Optional[str]:
        """Find a cached answer whose question is semantically close enough.

        Args:
            kb_version: Current knowledge base fingerprint (limits candidates to
                answers produced under the same data).
            query_embedding: Embedding of the incoming question.

        Returns:
            The best matching cached answer if its similarity meets the
            configured threshold, otherwise ``None``. Runs synchronously; call
            via ``asyncio.to_thread``.
        """
        candidates = self.db.get_semantic_candidates(kb_version, namespace="ask")
        best_answer: Optional[str] = None
        best_sim = 0.0
        best_id: Optional[int] = None
        for candidate in candidates:
            sim = cosine_similarity(query_embedding, candidate["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_answer = candidate["answer"]
                best_id = candidate["id"]

        if best_answer is not None and best_sim >= self.config.semantic_cache_threshold:
            if best_id is not None:
                self.db.register_cache_hit(best_id)
            self.log.info(
                "Semantic cache hit (similarity=%.3f, threshold=%.2f).",
                best_sim,
                self.config.semantic_cache_threshold,
            )
            return best_answer
        return None

    # -- message handling --------------------------------------------------

    def _strip_bot_mention(self, content: str) -> str:
        """Remove this bot's @mention tokens from message content."""
        if self.user is None:
            return content
        return re.sub(rf"<@!?{self.user.id}>", "", content)

    async def on_message(self, message: discord.Message) -> None:
        """Route messages to commands, treating a bare mention as a question.

        Behaviour:
        - ``!promo`` / ``@Bot promo`` and other explicit commands run as usual.
        - ``@Bot <free text>`` with no matching command is treated as ``!ask``,
          so users can simply mention the bot and ask a question naturally.

        Args:
            message: The incoming Discord message.
        """
        if message.author.bot:
            return  # Ignore other bots and our own messages.

        if not self.is_allowed_context(message):
            return  # Outside the allowed channels / DMs are disabled.

        ctx = await self.get_context(message)
        if ctx.command is not None:
            await self.invoke(ctx)
            return

        # No command matched. If the bot was directly mentioned, interpret the
        # remaining text as an !ask question.
        if self.user in message.mentions and not message.mention_everyone:
            question = self._strip_bot_mention(message.content).strip()
            if question:
                ask_command = self.get_command("ask")
                if ask_command is not None:
                    await ctx.invoke(ask_command, question=question)

    # -- lifecycle ---------------------------------------------------------

    async def on_ready(self) -> None:
        """Run startup verification checks once the bot has connected."""
        self.log.info("Successfully logged in as %s", self.user)

        # Verify database connectivity.
        try:
            await asyncio.to_thread(self.db.verify_connection)
            self.log.info("✓ Database connection OK.")
        except Exception:  # noqa: BLE001
            self.log.exception("✗ Database connection failed.")

        # Verify the Anthropic API key with a tiny request.
        try:
            await self.claude.messages.create(
                model=self.config.anthropic_model,
                max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
            )
            self.log.info("✓ Anthropic API key OK.")
        except Exception:  # noqa: BLE001
            self.log.exception("✗ Anthropic API verification failed.")

        # Verify connectivity to the validation source (internal only).
        try:
            reachable = await asyncio.to_thread(self.scraper.check_connection)
            if reachable:
                self.log.info("✓ Validation source reachable.")
            else:
                self.log.warning("✗ Validation source not reachable.")
        except Exception:  # noqa: BLE001
            self.log.exception("✗ Validation source check failed.")

        # Verify the semantic cache embedding backend, if configured.
        if self.semantic_enabled():
            try:
                ok = await asyncio.to_thread(self.embedder.health_check)
                if ok:
                    self.log.info("✓ Semantic cache (Voyage embeddings) OK.")
                else:
                    self.log.warning(
                        "✗ Voyage embeddings unreachable; semantic cache "
                        "inactive (exact-match cache still works)."
                    )
            except Exception:  # noqa: BLE001
                self.log.exception("✗ Voyage embeddings check failed.")
        else:
            self.log.info(
                "Semantic cache disabled (no VOYAGE_API_KEY); exact-match "
                "cache still active."
            )

        # Report FAQ knowledge base status.
        if self.faq and self.faq.version != "none":
            self.log.info(
                "✓ FAQ knowledge base loaded (version %s).", self.faq.version
            )
        else:
            self.log.info("FAQ knowledge base disabled or unavailable.")

        self.log.info("🚀 TickShift AI bot is ready.")

    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError
    ) -> None:
        """Global command error handler providing friendly Discord messages."""
        if isinstance(error, commands.CommandNotFound):
            return  # Ignore unknown commands silently.
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(
                f"⚠️ Missing input. Try `{self.config.command_prefix}help` "
                "to see how to use that command."
            )
            return
        self.log.exception("Unhandled command error: %s", error)
        await ctx.send(
            "😕 Something went wrong while handling that. Please try again."
        )

    # -- Claude helpers ----------------------------------------------------

    async def ask_claude(
        self,
        system_prompt: Union[str, List[Dict[str, object]]],
        user_message: str,
    ) -> str:
        """Send a single-turn message to Claude and return its text reply.

        Args:
            system_prompt: The dynamic system prompt — either a plain string or
                a list of system content blocks (which may carry
                ``cache_control`` markers for prompt caching).
            user_message: The user's message content.

        Returns:
            Claude's text response.

        Raises:
            anthropic.AnthropicError: If the API call fails.
        """
        response = await self.claude.messages.create(
            model=self.config.anthropic_model,
            max_tokens=self.config.anthropic_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.log.debug(
                "Claude usage: input=%s cache_read=%s cache_write=%s output=%s",
                getattr(usage, "input_tokens", "?"),
                getattr(usage, "cache_read_input_tokens", "?"),
                getattr(usage, "cache_creation_input_tokens", "?"),
                getattr(usage, "output_tokens", "?"),
            )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

    async def extract_firm_names(self, question: str) -> Tuple[List[str], bool]:
        """Use Claude to extract prop firm names mentioned in a question.

        Args:
            question: The user's free-text question.

        Returns:
            A tuple ``(firm_names, ambiguous)`` where ``firm_names`` is a list of
            explicitly named firms and ``ambiguous`` indicates the question
            references a firm without naming it (e.g. "that firm").
        """
        extraction_system = (
            "You extract proprietary trading firm names from a user's question. "
            "Respond with ONLY a compact JSON object of the form "
            '{"firms": ["Name1", "Name2"], "ambiguous": false}. '
            '"firms" lists the specific prop firm names explicitly mentioned '
            "(empty if none). Set \"ambiguous\" to true only if the user clearly "
            "refers to a specific firm without naming it (e.g. 'that firm'). "
            "Do not include generic words. Output JSON only, no prose."
        )
        try:
            raw = await self.ask_claude(extraction_system, question)
            return self._parse_extraction(raw)
        except Exception:  # noqa: BLE001 - extraction is best-effort.
            self.log.exception("Firm name extraction failed; assuming none.")
            return [], False

    @staticmethod
    def _parse_extraction(raw: str) -> Tuple[List[str], bool]:
        """Parse the JSON returned by the extraction prompt defensively."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return [], False
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], False
        firms = data.get("firms", []) or []
        if not isinstance(firms, list):
            firms = []
        cleaned = [str(f).strip() for f in firms if str(f).strip()]
        ambiguous = bool(data.get("ambiguous", False))
        return cleaned, ambiguous

    # -- knowledge base helpers -------------------------------------------

    async def build_prompt(
        self,
        extra_instructions: Optional[str] = None,
        question: Optional[str] = None,
    ) -> "PromptBundle":
        """Fetch current DB state and build a fresh, cache-friendly prompt.

        The prompt is split into two system blocks:

        1. A **stable** block (instructions + firm catalogue + help-center
           links) marked with ``cache_control`` so Anthropic prompt-caches it —
           repeat requests read this prefix at ~10% of the normal input price.
        2. An optional **varying** block of FAQ excerpts selected for this
           specific question. It sits after the cache breakpoint, so it never
           invalidates the cached prefix.

        Args:
            extra_instructions: Optional task-specific guidance to include.
            question: The user's question; when given (and the FAQ is enabled),
                relevant official-FAQ excerpts are retrieved and appended.

        Returns:
            A :class:`PromptBundle` with the system blocks and the knowledge
            fingerprint used by the answer cache.
        """
        markets = self.allowed_markets()
        include = self.included_firms()
        firms = await asyncio.to_thread(self.db.get_all_firms, markets, include)
        promos = await asyncio.to_thread(
            self.db.get_active_promo_codes, markets, include
        )
        help_centers = (
            self.faq.help_center_lines(markets, include) if self.faq else None
        )
        stable = build_system_prompt(
            firms, promos, extra_instructions, help_centers
        )

        system: List[Dict[str, object]] = [
            {
                "type": "text",
                "text": stable,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        if self.faq and question:
            excerpts = await asyncio.to_thread(
                self.faq.select_excerpts,
                question,
                markets,
                include,
                self.config.faq_max_chars,
            )
            if excerpts:
                system.append({"type": "text", "text": build_faq_block(excerpts)})

        # The answer-cache fingerprint covers the stable knowledge plus the FAQ
        # version — NOT the per-question excerpt — so cached answers stay valid
        # until the underlying data or FAQ file actually changes.
        faq_version = self.faq.version if self.faq else "off"
        fingerprint = kb_fingerprint(f"{stable}|faq:{faq_version}")
        return PromptBundle(system=system, fingerprint=fingerprint)

    async def resolve_firm_pair(self, raw: str) -> Optional[Tuple[str, str]]:
        """Resolve two firm names from free text using known names and aliases.

        Handles multi-word names without quotes (``my funded futures apex``),
        common aliases/misspellings (``mff``, ``tpt``, ``funded next``) and any
        separator (``vs``, comma, or just a space), by scanning the text for
        known firm names instead of guessing where one name ends.

        Args:
            raw: The raw argument text after the compare command.

        Returns:
            The two firm display names in order of appearance, or ``None`` if
            the text doesn't contain exactly two known firms.
        """
        low = raw.lower()
        if not low.strip():
            return None

        # Known display names from the database (all firms, so hidden ones
        # still resolve and get the proper "futures only" reply downstream)...
        firms = await asyncio.to_thread(self.db.get_all_firms)
        alias_map: Dict[str, str] = {
            f["name"].lower(): str(f["name"]) for f in firms if f.get("name")
        }
        # ...plus the FAQ aliases (mff, tpt, funded next, ...) mapped to the
        # section's display name.
        if self.faq:
            for alias, key in FIRM_ALIASES.items():
                section = self.faq._sections.get(key)  # noqa: SLF001
                if section and section.get("name"):
                    alias_map.setdefault(alias, str(section["name"]))

        # Find all alias occurrences; prefer longer aliases, dedupe by firm.
        hits: List[Tuple[int, int, str]] = []
        for alias, display in alias_map.items():
            pos = low.find(alias)
            if pos >= 0:
                hits.append((pos, -len(alias), display))
        hits.sort()
        ordered: List[str] = []
        for _, _, display in hits:
            if display not in ordered:
                ordered.append(display)
        if len(ordered) == 2:
            return ordered[0], ordered[1]
        return None

    async def validate_firms(
        self, firm_names: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Validate and ingest a list of firms.

        Args:
            firm_names: Firm names to validate.

        Returns:
            A tuple ``(all_valid, message)``. When ``all_valid`` is ``False``,
            ``message`` is a natural, user-facing explanation.
        """
        for name in firm_names:
            result = await asyncio.to_thread(
                self.validator.validate_and_ingest, name
            )
            if not result.is_valid:
                return False, result.message
        return True, None


# -- answer cache helpers --------------------------------------------------


def normalize_question(question: str) -> str:
    """Normalise a question for cache matching.

    Lowercases, collapses whitespace and strips surrounding punctuation so that
    trivially different phrasings of the same question share a cache key.

    Args:
        question: The raw question text.

    Returns:
        A normalised form suitable for hashing.
    """
    normalized = question.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" ?!.,")


def cache_key(prefix: str, text: str) -> str:
    """Return a stable SHA-256 hash for a namespaced cache key.

    Args:
        prefix: A namespace (e.g. ``"ask"`` or ``"compare"``) so different
            command types never collide.
        text: The already-normalised text to hash.
    """
    return hashlib.sha256(f"{prefix}:{text}".encode("utf-8")).hexdigest()


def kb_fingerprint(system_prompt: str) -> str:
    """Return a fingerprint of the knowledge base for cache invalidation.

    The system prompt is a faithful serialisation of the current firm/promo
    data, so hashing it yields a version that changes whenever that data does.

    Args:
        system_prompt: The dynamic system prompt built from the database.
    """
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


# -- message chunking utilities -------------------------------------------


def chunk_text(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> List[str]:
    """Split text into chunks that respect Discord's message length limit.

    Splitting prefers paragraph and line boundaries, and falls back to hard
    slicing for individual lines that exceed the limit.

    Args:
        text: The text to split.
        limit: Maximum characters per chunk.

    Returns:
        A list of message-sized chunks (never empty).
    """
    text = text.strip()
    if not text:
        return ["(no content)"]
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    current = ""
    for line in text.split("\n"):
        # Hard-split any single line that is itself too long.
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]

        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


async def send_chunked(ctx: commands.Context, text: str) -> None:
    """Send potentially long text to Discord across multiple messages."""
    for chunk in chunk_text(text):
        await ctx.send(chunk)


# -- command registration --------------------------------------------------


def register_commands(bot: TickShiftBot) -> None:
    """Register all bot commands on the given bot instance."""

    async def rate_limited(ctx: commands.Context) -> bool:
        """Return True (and notify the user) if this call is rate limited."""
        retry = bot.check_rate_limit(ctx.message)
        if retry:
            await ctx.send(
                f"⏳ Easy there — you're sending commands too fast. "
                f"Try again in {retry:.0f}s."
            )
            return True
        return False

    @bot.command(name="ask")
    async def ask(ctx: commands.Context, *, question: str = "") -> None:
        """Answer a free-text question about prop firms.

        Usage: ``!ask <question>``
        """
        if await rate_limited(ctx):
            return

        question = question.strip()
        if not question:
            await ctx.send(
                f"Please include a question, e.g. "
                f"`{bot.config.command_prefix}ask Which firm has the fastest payouts?`"
            )
            return

        if len(question) > bot.config.max_question_length:
            await ctx.send(
                f"That question is a bit long. Please keep it under "
                f"{bot.config.max_question_length} characters."
            )
            return

        async with ctx.typing():
            try:
                qhash = cache_key("ask", normalize_question(question))
                query_embedding: Optional[List[float]] = None

                if bot.config.qa_cache_enabled:
                    current_kb = (await bot.build_prompt()).fingerprint

                    # 1) Exact match — cheapest, no embedding or Claude call.
                    cached = await asyncio.to_thread(
                        bot.db.get_cached_answer,
                        qhash,
                        current_kb,
                        bot.config.qa_cache_ttl_days,
                    )
                    if cached:
                        await send_chunked(ctx, cached)
                        return

                    # 2) Semantic match — reuse an answer to a differently-worded
                    #    but equivalent question (embedding is cheap vs. Claude).
                    if bot.semantic_enabled():
                        query_embedding = await asyncio.to_thread(
                            bot.embedder.embed, question, "query"
                        )
                        if query_embedding:
                            hit = await asyncio.to_thread(
                                bot.semantic_lookup, current_kb, query_embedding
                            )
                            if hit:
                                await send_chunked(ctx, hit)
                                return

                firm_names, ambiguous = await bot.extract_firm_names(question)

                # If a firm is referenced but not named, ask for clarification.
                if ambiguous and not firm_names:
                    await ctx.send(
                        "Which prop firm are you asking about? "
                        "Let me know the name and I'll take a look."
                    )
                    return

                # Validate any explicitly named firms before answering.
                if firm_names:
                    all_valid, message = await bot.validate_firms(firm_names)
                    if not all_valid:
                        await ctx.send(message)
                        return

                # Rebuild the prompt (validation may have ingested new firms),
                # retrieving official-FAQ excerpts relevant to this question.
                bundle = await bot.build_prompt(question=question)
                answer = await bot.ask_claude(
                    bundle.system, wrap_user_question(question)
                )

                # Cache the fresh answer (with its embedding) against the KB
                # that produced it, so both exact and semantic hits work later.
                if bot.config.qa_cache_enabled and answer:
                    await asyncio.to_thread(
                        bot.db.store_cached_answer,
                        qhash,
                        question,
                        answer,
                        bundle.fingerprint,
                        query_embedding,
                        "ask",
                    )

                await send_chunked(ctx, answer or "I couldn't find an answer to that.")
            except anthropic.AnthropicError:
                bot.log.exception("Anthropic API error in !ask.")
                await ctx.send(
                    "😕 I'm having trouble reaching my brain right now. "
                    "Please try again in a moment."
                )
            except Exception:  # noqa: BLE001
                bot.log.exception("Unexpected error in !ask.")
                await ctx.send(
                    "😕 Something went wrong while answering. Please try again."
                )

    @bot.command(name="compare")
    async def compare(ctx: commands.Context, *, firms: str = "") -> None:
        """Compare two prop firms side by side.

        Usage: ``!compare <firm1> <firm2>``  (supports quotes and ``vs``)
        """
        if await rate_limited(ctx):
            return

        # Prefer known-name/alias resolution (handles multi-word names without
        # quotes, "mff", "tpt", misspellings); fall back to token splitting.
        resolved = await bot.resolve_firm_pair(firms)
        if resolved:
            firm1, firm2 = resolved
        else:
            firm1, firm2 = _parse_two_firms(firms)
        if not firm1 or not firm2:
            await ctx.send(
                f"Please name two firms to compare, e.g. "
                f"`{bot.config.command_prefix}compare Tradeify Topstep`."
            )
            return

        if len(firm1) > 100 or len(firm2) > 100:
            await ctx.send("Those firm names are too long.")
            return

        async with ctx.typing():
            try:
                # Order-independent cache key so "A vs B" and "B vs A" share it.
                pair = " | ".join(
                    sorted([firm1.strip().lower(), firm2.strip().lower()])
                )
                qhash = cache_key("compare", pair)

                if bot.config.qa_cache_enabled:
                    current_kb = (
                        await bot.build_prompt(COMPARE_INSTRUCTIONS)
                    ).fingerprint
                    cached = await asyncio.to_thread(
                        bot.db.get_cached_answer,
                        qhash,
                        current_kb,
                        bot.config.qa_cache_ttl_days,
                    )
                    if cached:
                        await _send_comparison_embed(ctx, firm1, firm2, cached)
                        return

                all_valid, message = await bot.validate_firms([firm1, firm2])
                if not all_valid:
                    # Use a comparison-specific natural message.
                    await ctx.send(
                        "I don't have enough information about one of those "
                        "firms to make a comparison."
                    )
                    return

                # In futures-only mode, refuse to compare a hidden (non-futures)
                # firm even though it validated as a real firm.
                markets = bot.allowed_markets()
                if markets:
                    include = bot.included_firms()
                    for name in (firm1, firm2):
                        in_scope = await asyncio.to_thread(
                            bot.db.get_firm_by_name, name, markets, include
                        )
                        if not in_scope:
                            await ctx.send(
                                "I focus on futures prop firms, so I can't "
                                "compare that pairing."
                            )
                            return

                # Retrieve FAQ excerpts for both firms via a synthetic question.
                bundle = await bot.build_prompt(
                    COMPARE_INSTRUCTIONS,
                    question=f"compare {firm1} vs {firm2} rules pricing payouts",
                )
                answer = await bot.ask_claude(
                    bundle.system,
                    f"Compare {firm1} and {firm2} for a trader deciding between them.",
                )

                if bot.config.qa_cache_enabled and answer:
                    await asyncio.to_thread(
                        bot.db.store_cached_answer,
                        qhash,
                        f"compare {firm1} vs {firm2}",
                        answer,
                        bundle.fingerprint,
                        None,
                        "compare",
                    )

                await _send_comparison_embed(ctx, firm1, firm2, answer)
            except anthropic.AnthropicError:
                bot.log.exception("Anthropic API error in !compare.")
                await ctx.send(
                    "😕 I'm having trouble generating that comparison right now."
                )
            except Exception:  # noqa: BLE001
                bot.log.exception("Unexpected error in !compare.")
                await ctx.send(
                    "😕 Something went wrong while comparing those firms."
                )

    @bot.command(name="firm")
    async def firm(ctx: commands.Context, *, firm_name: str = "") -> None:
        """Show a detailed profile for a single firm.

        Usage: ``!firm <firm_name>``
        """
        if await rate_limited(ctx):
            return

        firm_name = firm_name.strip().strip('"').strip("'")
        if not firm_name:
            await ctx.send(
                f"Please name a firm, e.g. "
                f"`{bot.config.command_prefix}firm Tradeify`."
            )
            return

        if len(firm_name) > 100:
            await ctx.send("That firm name is too long.")
            return

        async with ctx.typing():
            try:
                result = await asyncio.to_thread(
                    bot.validator.validate_and_ingest, firm_name
                )
                if not result.is_valid:
                    await ctx.send(
                        "That firm isn't well-known enough for me to have "
                        "detailed information."
                    )
                    return

                data = await asyncio.to_thread(
                    bot.db.get_firm_by_name,
                    firm_name,
                    bot.allowed_markets(),
                    bot.included_firms(),
                )
                if not data:
                    # Either unknown, or a non-futures firm hidden in this mode.
                    if bot.config.futures_only:
                        await ctx.send(
                            "I focus on futures prop firms, so I don't have a "
                            "profile for that one."
                        )
                    else:
                        await ctx.send(
                            "I don't have detailed data on that firm right now."
                        )
                    return
                await ctx.send(embed=_build_firm_embed(data))
            except Exception:  # noqa: BLE001
                bot.log.exception("Unexpected error in !firm.")
                await ctx.send(
                    "😕 Something went wrong while fetching that firm."
                )

    @bot.command(name="promo")
    async def promo(ctx: commands.Context) -> None:
        """List all active promo codes across every firm.

        Usage: ``!promo``
        """
        if await rate_limited(ctx):
            return
        async with ctx.typing():
            try:
                promos = await asyncio.to_thread(
                    bot.db.get_active_promo_codes,
                    bot.allowed_markets(),
                    bot.included_firms(),
                )
                if not promos:
                    await ctx.send("There are no active promo codes right now.")
                    return
                await ctx.send(embed=_build_promo_embed(promos))
            except Exception:  # noqa: BLE001
                bot.log.exception("Unexpected error in !promo.")
                await ctx.send(
                    "😕 Something went wrong while fetching promo codes."
                )

    @bot.command(name="help")
    async def help_command(ctx: commands.Context) -> None:
        """Show all available commands and how to use them.

        Usage: ``!help``
        """
        prefix = bot.config.command_prefix
        mention = bot.user.mention if bot.user else "@TickShift"
        embed = discord.Embed(
            title="TickShift AI — Commands",
            description=(
                "Your assistant for everything about prop trading firms.\n"
                f"Tip: you can use `{prefix}` **or** just mention me — "
                f"e.g. `{mention} which firm has the fastest payouts?`"
            ),
            color=discord.Color.blurple(),
        )
        embed.add_field(
            name=f"{prefix}ask <question>",
            value=(
                "Ask anything about prop firms.\n"
                f"e.g. `{prefix}ask Which firm has the fastest payouts?`"
            ),
            inline=False,
        )
        embed.add_field(
            name=f"{prefix}compare <firm1> <firm2>",
            value=(
                "Compare two firms side by side.\n"
                f"e.g. `{prefix}compare FTMO Tradeify`"
            ),
            inline=False,
        )
        embed.add_field(
            name=f"{prefix}firm <firm_name>",
            value=(
                "Show a detailed profile for a firm.\n"
                f"e.g. `{prefix}firm Tradeify`"
            ),
            inline=False,
        )
        embed.add_field(
            name=f"{prefix}promo",
            value="List all currently active promo codes.",
            inline=False,
        )
        embed.add_field(
            name=f"{prefix}help",
            value="Show this help message.",
            inline=False,
        )
        embed.set_footer(
            text="TickShift AI • Powered by Claude • Use ! or @mention me"
        )
        await ctx.send(embed=embed)


# -- embed / parsing helpers ----------------------------------------------


def _parse_two_firms(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse two firm names from a compare argument string.

    Supports quoted names, a ``vs``/``versus`` separator, or two space-separated
    tokens.

    Args:
        raw: The raw argument text after ``!compare``.

    Returns:
        A tuple of two firm names, either of which may be ``None``.
    """
    raw = raw.strip()
    if not raw:
        return None, None

    # Quoted names take priority: "Firm One" "Firm Two".
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', raw)
    names = [a or b for a, b in quoted]
    if len(names) >= 2:
        return names[0].strip(), names[1].strip()

    # Explicit separator: vs / versus / comma.
    for separator in (r"\bvs\.?\b", r"\bversus\b", ","):
        parts = re.split(separator, raw, flags=re.IGNORECASE)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return parts[0].strip(), parts[1].strip()

    # Fallback: split into two words/tokens.
    tokens = raw.split()
    if len(tokens) >= 2:
        # Assume single-word firm names; put the split in the middle otherwise.
        if len(tokens) == 2:
            return tokens[0], tokens[1]
        mid = len(tokens) // 2
        return " ".join(tokens[:mid]), " ".join(tokens[mid:])
    return None, None


def _build_firm_embed(data: Dict[str, object]) -> discord.Embed:
    """Build a rich Discord embed describing a single firm."""
    details = data.get("details", []) or []
    platforms = [
        str(d.get("value"))
        for d in details
        if d.get("type") == "trading_platform" and d.get("value")
    ]
    payout_methods = [
        str(d.get("value"))
        for d in details
        if d.get("type") == "payout_method" and d.get("value")
    ]
    promos = data.get("promo_codes", []) or []

    color = (
        discord.Color.red()
        if data.get("warning_flag")
        else discord.Color.green()
    )
    embed = discord.Embed(
        title=str(data.get("name", "Unknown Firm")),
        description=(data.get("description") or "")[:EMBED_DESCRIPTION_LIMIT] or None,
        color=color,
    )

    embed.add_field(name="Tier", value=str(data.get("tier") or "N/A"), inline=True)
    embed.add_field(
        name="Country", value=str(data.get("country") or "N/A"), inline=True
    )
    embed.add_field(
        name="Founded", value=str(data.get("founded_year") or "N/A"), inline=True
    )

    max_alloc = data.get("max_allocation")
    embed.add_field(
        name="Max Allocation",
        value=f"${int(max_alloc):,}" if max_alloc is not None else "N/A",
        inline=True,
    )
    split = data.get("profit_split")
    embed.add_field(
        name="Profit Split",
        value=f"{split}%" if split is not None else "N/A",
        inline=True,
    )
    fee = data.get("challenge_fee_from")
    embed.add_field(
        name="Challenge Fee From",
        value=f"${float(fee):,.2f}" if fee is not None else "N/A",
        inline=True,
    )

    embed.add_field(
        name="Payout Frequency",
        value=str(data.get("payout_frequency") or "N/A"),
        inline=False,
    )
    if platforms:
        embed.add_field(
            name="Trading Platforms", value=", ".join(platforms), inline=False
        )
    if payout_methods:
        embed.add_field(
            name="Payout Methods", value=", ".join(payout_methods), inline=False
        )

    if promos:
        promo_lines = []
        for promo in promos:
            pct = promo.get("discount_percentage")
            pct_str = f" — {pct}% off" if pct is not None else ""
            promo_lines.append(f"`{promo.get('code')}`{pct_str}")
        embed.add_field(
            name="Active Promo Codes",
            value="\n".join(promo_lines),
            inline=False,
        )

    if data.get("warning_flag"):
        embed.add_field(
            name="⚠️ Warning",
            value=str(data.get("warning_message") or "Exercise caution."),
            inline=False,
        )

    embed.set_footer(text="TickShift AI")
    return embed


def _build_promo_embed(promos: List[Dict[str, object]]) -> discord.Embed:
    """Build a Discord embed listing all active promo codes."""
    embed = discord.Embed(
        title="🎟️ Active Promo Codes",
        description="Current discount codes across all tracked firms.",
        color=discord.Color.gold(),
    )
    for promo in promos[:25]:  # Discord embeds allow at most 25 fields.
        pct = promo.get("discount_percentage")
        pct_str = f"{pct}% off" if pct is not None else "Discount"
        desc = promo.get("description")
        value = f"`{promo.get('code')}` — {pct_str}"
        if desc:
            value += f"\n{desc}"
        embed.add_field(
            name=str(promo.get("firm_name", "Firm")),
            value=value,
            inline=False,
        )
    if len(promos) > 25:
        embed.set_footer(text=f"Showing 25 of {len(promos)} promo codes.")
    else:
        embed.set_footer(text="TickShift AI")
    return embed


async def _send_comparison_embed(
    ctx: commands.Context, firm1: str, firm2: str, answer: str
) -> None:
    """Send a comparison result as an embed, chunking if it is too long."""
    answer = answer.strip() or "I couldn't generate a comparison."
    title = f"⚖️ {firm1} vs {firm2}"
    if len(answer) <= EMBED_DESCRIPTION_LIMIT:
        embed = discord.Embed(
            title=title,
            description=answer,
            color=discord.Color.blurple(),
        )
        embed.set_footer(text="TickShift AI")
        await ctx.send(embed=embed)
        return

    # Long comparisons: first chunk in an embed, remainder as follow-up messages.
    chunks = chunk_text(answer, EMBED_DESCRIPTION_LIMIT)
    embed = discord.Embed(
        title=title, description=chunks[0], color=discord.Color.blurple()
    )
    embed.set_footer(text="TickShift AI")
    await ctx.send(embed=embed)
    for chunk in chunks[1:]:
        await send_chunked(ctx, chunk)


# -- bootstrap -------------------------------------------------------------


def main() -> None:
    """Load configuration, wire up dependencies and run the bot."""
    try:
        config = Config.load()
    except ConfigError as exc:
        # Logging may not be configured yet; write directly to stderr.
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    logger = configure_logging(config)
    logger.info("Starting TickShift AI Discord bot...")

    db = Database(config.database_url, refresh_days=config.search_refresh_days)
    try:
        db.init()
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to initialise the database. Check DATABASE_URL and that "
            "PostgreSQL is running."
        )
        sys.exit(1)

    # Seed the curated TickShift firm catalogue (idempotent upserts).
    if config.seed_on_startup:
        try:
            seed_database(db)
        except Exception:  # noqa: BLE001 - seeding must never block startup.
            logger.exception("Failed to seed the TickShift catalogue.")

    scraper = PropFirmScraper(
        base_url=config.propfirmmatch_base_url,
        min_interval=config.scrape_min_interval,
    )
    validator = FirmValidator(db, scraper)

    # Optionally set up semantic (meaning-based) answer caching via Voyage.
    embedder: Optional[EmbeddingClient] = None
    if config.semantic_cache_enabled and config.voyage_api_key:
        try:
            embedder = EmbeddingClient(config.voyage_api_key, config.voyage_model)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Could not initialise Voyage embeddings; semantic caching "
                "disabled (exact-match caching still active)."
            )
            embedder = None
    elif config.semantic_cache_enabled:
        logger.info(
            "VOYAGE_API_KEY not set; semantic caching disabled "
            "(exact-match caching still active)."
        )

    bot = TickShiftBot(config, db, scraper, validator, logger, embedder)
    register_commands(bot)

    try:
        bot.run(config.discord_token, log_handler=None)
    except discord.LoginFailure:
        logger.error("Invalid DISCORD_TOKEN — the bot could not log in.")
        sys.exit(1)
    except Exception:  # noqa: BLE001
        logger.exception("The bot crashed unexpectedly.")
        sys.exit(1)
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
