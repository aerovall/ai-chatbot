# TickShift AI — Discord Chatbot

A production-ready Discord chatbot that answers questions about proprietary
("prop") trading firms using **Claude AI** and a **dynamic, self-updating
knowledge base** backed by PostgreSQL.

When a user asks about a firm the bot doesn't know yet, it validates the firm,
collects information about it, stores it in the database, and answers — all
without a code change or redeploy. Every response is built from the *current*
contents of the database, so the knowledge base stays live.

---

## Features

| Command | Description |
| --- | --- |
| `!ask <question>` | Ask anything about prop firms. Any firm you name is validated and, if legitimate, researched and added to the knowledge base before the bot answers. |
| `!compare <firm1> <firm2>` | Side-by-side comparison of two firms, returned as a Discord embed. |
| `!firm <firm_name>` | A detailed profile of a single firm as a rich embed (tier, country, allocation, fees, payouts, platforms, promo codes, warnings). |
| `!promo` | Lists all active promo codes across every tracked firm. |
| `!help` | Shows all commands and usage. |

Additional capabilities:

- **Dynamic system prompt** — firm data and promo codes are fetched from the
  database on every query and injected into Claude's system prompt.
- **Automatic firm ingestion** — legitimate firms that aren't in the database
  yet are researched and stored on demand.
- **Search caching & refresh** — a `firm_search_cache` table avoids repeated
  lookups and refreshes stale entries (default: every 30 days).
- **Answer caching** — repeat `!ask`/`!compare` questions are served from a
  `qa_cache` table instead of calling Claude again, saving API credits. Cached
  answers are fingerprinted against the knowledge base, so they're automatically
  regenerated whenever firm or promo data changes.
- **Semantic caching** — with a Voyage AI key, `!ask` also matches
  differently-worded questions that mean the same thing (e.g. "fastest payouts"
  ≈ "quickest withdrawals") using embeddings, further reducing Claude calls.
  Falls back to exact-match caching if no key is configured.
- **Long-message handling** — responses are chunked to respect Discord's
  2000-character limit.
- **Robust error handling & logging** — friendly Discord messages, full logs to
  a rotating file, and graceful degradation when a dependency is unavailable.
- **Polite scraping** — shared rate limiter, descriptive User-Agent, retries
  with backoff, and `robots.txt` awareness.

---

## Project structure

```
tickshift-discord-bot/
├── .env.example        # Template for environment variables
├── requirements.txt    # Python dependencies
├── bot.py              # Discord bot entry point & commands
├── config.py           # Configuration & logging setup
├── database.py         # SQLAlchemy models & query functions
├── prompts.py          # Dynamic system prompt builder
├── validator.py        # Firm validation & lazy ingestion
├── scraper.py          # Web scraping & data extraction
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **PostgreSQL 12+** (a reachable database and connection string)
- A **Discord bot token** — create an application at the
  [Discord Developer Portal](https://discord.com/developers/applications),
  add a Bot, and enable the **Message Content Intent** under *Bot → Privileged
  Gateway Intents*.
- An **Anthropic API key** from the
  [Anthropic Console](https://console.anthropic.com/).

---

## Setup

1. **Clone and enter the project**

   ```bash
   git clone <your-repo-url>
   cd tickshift-discord-bot
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create the PostgreSQL database**

   ```bash
   createdb tickshift_db
   ```

   The bot creates all required tables automatically on first run.

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env`:

   ```dotenv
   DISCORD_TOKEN=your_discord_bot_token
   ANTHROPIC_API_KEY=your_anthropic_api_key
   DATABASE_URL=postgresql://user:password@localhost:5432/tickshift_db
   PROPFIRMMATCH_BASE_URL=https://propfirmmatch.com/
   ```

   Optional tuning variables (model, rate limits, refresh window, logging) are
   documented in `.env.example`.

5. **Run the bot**

   ```bash
   python bot.py
   ```

   On startup the bot verifies its database connection, Anthropic API key and
   outbound connectivity, then logs `🚀 TickShift AI bot is ready.`

6. **Invite the bot to your server**

   In the Developer Portal, use *OAuth2 → URL Generator* with the `bot` scope
   and the *Send Messages*, *Embed Links* and *Read Message History*
   permissions, then open the generated URL.

---

## Usage examples

```
!ask Which firm has the fastest payouts?
!ask Tell me about Tradeify
!compare FTMO Tradeify
!compare "The Funded Trader" "Funding Pips"
!firm Tradeify
!promo
!help
```

You can also **@mention the bot** instead of using the `!` prefix:

```
@TickShift promo
@TickShift compare FTMO Tradeify
@TickShift which firm has the fastest payouts?
```

A bare mention followed by a plain question is treated as `!ask`, so users can
just tag the bot and ask naturally.

---

## Configuration reference

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `DISCORD_TOKEN` | ✅ | — | Discord bot token. |
| `ANTHROPIC_API_KEY` | ✅ | — | Anthropic API key. |
| `DATABASE_URL` | ✅ | — | PostgreSQL connection string. |
| `PROPFIRMMATCH_BASE_URL` | — | `https://propfirmmatch.com/` | Base URL used internally for firm validation. |
| `ANTHROPIC_MODEL` | — | `claude-sonnet-5` | Claude model used for answers. |
| `ANTHROPIC_MAX_TOKENS` | — | `1024` | Max tokens per Claude response. |
| `COMMAND_PREFIX` | — | `!` | Command prefix. |
| `SCRAPE_MIN_INTERVAL` | — | `2.0` | Minimum seconds between scraping requests. |
| `SEARCH_REFRESH_DAYS` | — | `30` | Days before a cached firm search is refreshed. |
| `SEED_ON_STARTUP` | — | `true` | Seed the curated TickShift firm catalogue (firms + promo codes) on startup. |
| `QA_CACHE_ENABLED` | — | `true` | Cache Claude answers so repeat questions don't re-hit the API. |
| `QA_CACHE_TTL_DAYS` | — | `0` | Max age (days) for a cached answer; `0` = invalidate only on data change. |
| `SEMANTIC_CACHE_ENABLED` | — | `true` | Enable meaning-based matching of questions (needs `VOYAGE_API_KEY`). |
| `VOYAGE_API_KEY` | — | — | Voyage AI key for semantic caching. Omit to use exact-match caching only. |
| `VOYAGE_MODEL` | — | `voyage-3.5` | Voyage embedding model. |
| `SEMANTIC_CACHE_THRESHOLD` | — | `0.85` | Cosine-similarity threshold for treating two questions as equivalent. |
| `LOG_LEVEL` | — | `INFO` | Logging verbosity. |
| `LOG_FILE` | — | `logs/bot.log` | Log file path. |

---

## Database schema

Four tables are created automatically:

- **`prop_firms`** — headline firm data (name, tier, country, allocation,
  profit split, fees, payout frequency, description, warnings).
- **`firm_details`** — flexible key/value attributes (trading platforms, payout
  methods, trading rules).
- **`promo_codes`** — active discount codes per firm.
- **`firm_search_cache`** — records which firms have been validated/searched and
  when, powering the caching and refresh logic.
- **`qa_cache`** — caches Claude answers to repeat questions, fingerprinted
  against the knowledge base so stale answers are regenerated automatically.

---

## How the dynamic knowledge base works

1. A user issues a command that references a firm.
2. The firm name is validated internally. Unrecognised firms receive a natural,
   varied "I'm not familiar with that firm" style response.
3. Recognised firms that aren't in the database yet are researched, structured
   and stored.
4. The bot fetches **all** current firm and promo data from the database and
   builds a fresh system prompt.
5. Claude answers using only that live knowledge base.

Because the prompt is rebuilt on every request, updates to the database take
effect immediately — no redeployment required.

### Seed catalogue

On startup the bot seeds a curated set of firms and their active promo codes
from the [TickShift catalogue](https://www.tickshift.app/) (see `seed.py`), so
commands like `!promo` and `!firm` return data immediately on a fresh database.
Seeding is idempotent (upsert by firm name) and can be disabled with
`SEED_ON_STARTUP=false`. You can also run it manually:

```bash
python seed.py
```

To adjust the catalogue or promo codes, edit `SEED_FIRMS` in `seed.py`.

---

## Extending the bot

Commands are registered in `register_commands()` in `bot.py`. To add a new
command, define an `async` function decorated with `@bot.command(...)` there;
reuse `bot.build_prompt()`, `bot.ask_claude()` and `bot.validate_firms()` for
consistent behaviour.

---

## Troubleshooting

- **`Configuration error: Missing required environment variable(s)`** — ensure
  `.env` exists and the required keys are set.
- **Bot logs in but ignores commands** — enable the **Message Content Intent**
  in the Developer Portal.
- **Database errors on startup** — confirm PostgreSQL is running and
  `DATABASE_URL` is correct; the database itself must already exist.
- **Detailed diagnostics** — set `LOG_LEVEL=DEBUG` and check `logs/bot.log`.

---

## Notes on responsible scraping

The bot honours `robots.txt`, rate-limits outbound requests, backs off on HTTP
`429` responses, and caches results to minimise load on external sites. Adjust
`SCRAPE_MIN_INTERVAL` upward if you need to be more conservative.
