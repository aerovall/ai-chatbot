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
import json
import logging
import re
import sys
from typing import Dict, List, Optional, Tuple

import anthropic
import discord
from discord.ext import commands

from config import Config, ConfigError, configure_logging
from database import Database
from prompts import COMPARE_INSTRUCTIONS, build_system_prompt
from scraper import PropFirmScraper
from validator import FirmValidator

DISCORD_MESSAGE_LIMIT = 2000
EMBED_DESCRIPTION_LIMIT = 4096


class TickShiftBot(commands.Bot):
    """A Discord bot that answers prop firm questions using Claude."""

    def __init__(
        self,
        config: Config,
        db: Database,
        scraper: PropFirmScraper,
        validator: FirmValidator,
        logger: logging.Logger,
    ) -> None:
        """Construct the bot and its collaborators.

        Args:
            config: Validated runtime configuration.
            db: Database helper.
            scraper: Prop firm scraper.
            validator: Firm validator.
            logger: Application logger.
        """
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read command text.
        super().__init__(
            command_prefix=config.command_prefix,
            intents=intents,
            help_command=None,  # We provide a custom !help.
        )
        self.config = config
        self.db = db
        self.scraper = scraper
        self.validator = validator
        self.log = logger
        self.claude = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

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
        self, system_prompt: str, user_message: str
    ) -> str:
        """Send a single-turn message to Claude and return its text reply.

        Args:
            system_prompt: The dynamic system prompt.
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
        self, extra_instructions: Optional[str] = None
    ) -> str:
        """Fetch current DB state and build a fresh system prompt.

        Args:
            extra_instructions: Optional task-specific guidance to include.

        Returns:
            A complete system prompt reflecting the live knowledge base.
        """
        firms = await asyncio.to_thread(self.db.get_all_firms)
        promos = await asyncio.to_thread(self.db.get_active_promo_codes)
        return build_system_prompt(firms, promos, extra_instructions)

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

    @bot.command(name="ask")
    async def ask(ctx: commands.Context, *, question: str = "") -> None:
        """Answer a free-text question about prop firms.

        Usage: ``!ask <question>``
        """
        question = question.strip()
        if not question:
            await ctx.send(
                f"Please include a question, e.g. "
                f"`{bot.config.command_prefix}ask Which firm has the fastest payouts?`"
            )
            return

        async with ctx.typing():
            try:
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

                system_prompt = await bot.build_prompt()
                answer = await bot.ask_claude(system_prompt, question)
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
        firm1, firm2 = _parse_two_firms(firms)
        if not firm1 or not firm2:
            await ctx.send(
                f"Please name two firms to compare, e.g. "
                f"`{bot.config.command_prefix}compare FTMO Tradeify`."
            )
            return

        async with ctx.typing():
            try:
                all_valid, message = await bot.validate_firms([firm1, firm2])
                if not all_valid:
                    # Use a comparison-specific natural message.
                    await ctx.send(
                        "I don't have enough information about one of those "
                        "firms to make a comparison."
                    )
                    return

                system_prompt = await bot.build_prompt(COMPARE_INSTRUCTIONS)
                answer = await bot.ask_claude(
                    system_prompt,
                    f"Compare {firm1} and {firm2} for a trader deciding between them.",
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
        firm_name = firm_name.strip().strip('"').strip("'")
        if not firm_name:
            await ctx.send(
                f"Please name a firm, e.g. "
                f"`{bot.config.command_prefix}firm Tradeify`."
            )
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
                    bot.db.get_firm_by_name, firm_name
                )
                if not data:
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
        async with ctx.typing():
            try:
                promos = await asyncio.to_thread(bot.db.get_active_promo_codes)
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
        embed = discord.Embed(
            title="TickShift AI — Commands",
            description="Your assistant for everything about prop trading firms.",
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
        embed.set_footer(text="TickShift AI • Powered by Claude")
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

    scraper = PropFirmScraper(
        base_url=config.propfirmmatch_base_url,
        min_interval=config.scrape_min_interval,
    )
    validator = FirmValidator(db, scraper)

    bot = TickShiftBot(config, db, scraper, validator, logger)
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
