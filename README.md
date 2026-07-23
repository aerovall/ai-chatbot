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
- **FAQ-first answering** — a bundled knowledge base of official help-center
  FAQ content (`data/faq_knowledge_base.json`; covers all nine
  futures firms — deep coverage for Tradeify, Take Profit Trader and Apex
  Trader Funding; summary coverage for Lucid Trading, Topstep, My Funded
  Futures, FundedNext, TradeDay and Alpha Futures — plus general prop-firm
  concepts and a quick reference) is searched per question with keyword retrieval (`faq.py`); only
  the relevant excerpts are injected into Claude's context. Answers cite the
  official source URL when available, never invent details, and point users to
  each firm's official help center for anything not covered.
- **Anthropic prompt caching** — the stable prompt prefix (instructions + firm
  catalogue + help-center links) carries a `cache_control` marker, so repeat
  requests read it at ~10% of the normal input-token price. The per-question
  FAQ excerpts sit after the cache breakpoint and never invalidate it.
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
| `ANTHROPIC_MAX_TOKENS` | — | `700` | Max tokens per Claude response. |
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
| `MAX_QUESTION_LENGTH` | — | `500` | Maximum characters allowed in an `!ask` question. |
| `RATE_LIMIT_PER_USER` | — | `5` | Max commands per user within `RATE_LIMIT_WINDOW`. |
| `RATE_LIMIT_WINDOW` | — | `60` | Rate-limit window in seconds. |
| `RATE_LIMIT_GLOBAL` | — | `30` | Max commands across all users within the window (credit-spend cap). |
| `ALLOWED_CHANNEL_IDS` | — | — | Comma-separated channel IDs the bot may respond in. Empty = all channels. |
| `IGNORE_DMS` | — | `false` | If `true`, the bot ignores direct messages (servers only). |
| `FUTURES_ONLY` | — | `true` | Serve/discuss only futures firms; hide forex/CFD & prediction firms (data retained). |
| `FAQ_ENABLED` | — | `true` | Retrieve official help-center FAQ excerpts into context per question. |
| `FAQ_MAX_CHARS` | — | `20000` | Character budget for FAQ excerpts per question. |
| `PROMPT_CACHE_TTL` | — | `1h` | Prompt-cache TTL for the stable prefix: `1h` or `5m`. |
| `INCLUDED_FIRMS` | — | `FXIFY` | Non-futures firms to keep listed (forex/CFD specifics stripped) in futures-only mode. |
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

On startup the bot seeds the full firm catalogue from the
[TickShift website](https://www.tickshift.app/) — every firm with its tier,
country, founding year, max allocation, profit split, cheapest challenge fee,
payout frequency, trading platforms, payout methods, trading rules, account
types, description and active promo codes. This means `!promo`, `!firm` and
`!ask` return rich data immediately on a fresh database.

The authoritative data is bundled as `data/tickshift_data.json` and transformed
into the database schema by `seed.py`. Seeding is idempotent (upsert by firm
name) and can be disabled with `SEED_ON_STARTUP=false`. You can also run it
manually:

```bash
python seed.py
```

To refresh the catalogue, replace `data/tickshift_data.json` with a newer export
and redeploy — no code changes required.

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

## Security & guardrails

The bot is built to be safe to run in a public server. It keeps users on-task
(asking questions) and can't easily be tampered with:

- **Prompt-injection / jailbreak resistance** — the system prompt hard-locks the
  bot to prop-firm topics and forbids revealing its instructions, changing its
  role, or following embedded commands. Every user message is passed to the model
  wrapped as clearly-delimited *untrusted* input, so attempts like "ignore your
  instructions" or "print your system prompt" are refused.
- **Futures-only knowledge base** — with `FUTURES_ONLY=true` (the default), the
  bot serves and discusses **only futures prop firms**. Forex/CFD and
  prediction-market firms (and forex/CFD account types on hybrid firms) stay in
  the database but are hidden from every answer, comparison, promo list and firm
  profile. The market classification mirrors the TickShift website
  (`funding-predicts` → predictions, `fxify`/`*(CFD)` account types → cfd, the
  rest → futures). Set `FUTURES_ONLY=false` to surface all markets again — no
  data is lost. Specific non-futures firms can be kept in the lineup via
  `INCLUDED_FIRMS` (default `FXIFY`): they stay listed with their promo code and
  general details, but their forex/CFD platforms, leverage and account specifics
  are stripped out.
- **Topic scope** — the bot answers questions about firms (programs, rules, fees,
  payouts, tiers, platforms, promos, comparisons) and declines trading advice —
  strategies, analysis, price predictions, signals, "how/what to trade" — always
  redirecting to futures prop-firm topics.
- **No mass pings** — the bot is configured with `allowed_mentions = none`, so it
  can never be tricked into pinging `@everyone`, `@here`, or roles, regardless of
  what a firm description or model response contains.
- **Rate limiting** — per-user and global limits (`RATE_LIMIT_*`) stop spam and
  cap how fast the bot can spend Anthropic credits. Over-limit users get a polite
  "slow down" message.
- **Input bounds** — questions and firm names are length-capped
  (`MAX_QUESTION_LENGTH`) to limit abuse and token cost.
- **Scope control** — restrict the bot to specific channels with
  `ALLOWED_CHANNEL_IDS`, and optionally disable DMs with `IGNORE_DMS`.
- **No SQL injection** — all database access uses parameterised SQLAlchemy
  queries; user text is never concatenated into SQL.
- **No admin/destructive commands** — the command surface is read-only
  (questions and lookups); there is nothing a user can invoke to mutate or wipe
  data.

### Recommended Discord & hosting hygiene

- **Grant the bot minimal permissions** — only *View Channel*, *Send Messages*,
  *Embed Links*, and *Read Message History*. It does **not** need Administrator,
  kick/ban, or message-management permissions.
- **Keep secrets in environment variables only** — never commit `.env`. Rotate
  any token that may have been exposed.
- **Restrict the invite** — add the bot only to servers you trust, and consider
  limiting it to a dedicated channel via `ALLOWED_CHANNEL_IDS`.
- **Least-privilege database user** — point `DATABASE_URL` at a user scoped to
  the bot's own database.

## Notes on responsible scraping

The bot honours `robots.txt`, rate-limits outbound requests, backs off on HTTP
`429` responses, and caches results to minimise load on external sites. Adjust
`SCRAPE_MIN_INTERVAL` upward if you need to be more conservative.
