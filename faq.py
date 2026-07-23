"""FAQ knowledge-base retrieval for the TickShift Discord bot.

Implements the FAQ-first design: a bundled knowledge base of official
help-center FAQ content (``data/faq_knowledge_base.json``) is searched with
lightweight keyword matching, and only the chunks relevant to the user's
question are injected into Claude's context. This keeps answers grounded in
official documentation while spending tokens only on what each question needs.

The knowledge base covers all nine futures firms — deep coverage for Tradeify,
Take Profit Trader and Apex Trader Funding; summary FAQ coverage for Lucid
Trading, Topstep, My Funded Futures, FundedNext, TradeDay and Alpha Futures —
plus general prop-firm concepts, a quick reference, and official help-center
links for every firm. Additional content can be appended to the JSON without
code changes.

Retrieval respects the bot's visibility rules: sections and chunks tied to
hidden firms (non-futures firms outside the include list) are never selected.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("tickshift.faq")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "faq_knowledge_base.json")

# Aliases users type -> canonical FAQ section key.
FIRM_ALIASES: Dict[str, str] = {
    "tradeify": "tradeify",
    "take profit trader": "take-profit-trader",
    "take profit": "take-profit-trader",
    "takeprofit": "take-profit-trader",
    "tpt": "take-profit-trader",
    "apex": "apex-trader-funding",
    "apex trader funding": "apex-trader-funding",
    "apex trader": "apex-trader-funding",
    "lucid": "lucid-trading",
    "lucid trading": "lucid-trading",
    "topstep": "topstep",
    "top step": "topstep",
    "my funded futures": "my-funded-futures",
    "my funded future": "my-funded-futures",
    "myfundedfutures": "my-funded-futures",
    "mff": "my-funded-futures",
    "fundednext": "fundednext",
    "funded next": "fundednext",
    "tradeday": "tradeday",
    "trade day": "tradeday",
    "traded day": "tradeday",
    "alpha futures": "alpha-futures",
    "alphafutures": "alpha-futures",
    "alpha": "alpha-futures",
}

# Official help centers for every firm (shown when the KB lacks deep coverage).
# Keyed by display name; the market tag drives visibility filtering.
HELP_CENTERS: List[Dict[str, str]] = [
    {"name": "Lucid Trading", "market": "futures", "url": "https://support.lucidtrading.com"},
    {"name": "My Funded Futures", "market": "futures", "url": "https://help.myfundedfutures.com"},
    {"name": "Topstep", "market": "futures", "url": "https://help.topstep.com"},
    {"name": "FundedNext", "market": "futures", "url": "https://helpfutures.fundednext.com"},
    {"name": "TradeDay", "market": "futures", "url": "https://tradeday.freshdesk.com"},
    {"name": "Alpha Futures", "market": "futures", "url": "https://help.alpha-futures.com"},
    {"name": "Tradeify", "market": "futures", "url": "https://help.tradeify.co/en/"},
    {"name": "Take Profit Trader", "market": "futures", "url": "https://help.takeprofittrader.com"},
    {"name": "Apex Trader Funding", "market": "futures", "url": "https://support.apextraderfunding.com"},
    {"name": "FXIFY", "market": "cfd", "url": "https://fxify.com/faqs"},
    {"name": "Funding Predicts", "market": "predictions", "url": "https://fundingpredicts.com"},
]

# Names of firms that may appear inside general chunks; used to drop chunks
# that reference firms hidden by the current visibility configuration.
_KNOWN_FIRM_NAMES: Dict[str, str] = {
    "lucid trading": "futures",
    "my funded futures": "futures",
    "topstep": "futures",
    "fundednext": "futures",
    "tradeday": "futures",
    "alpha futures": "futures",
    "tradeify": "futures",
    "take profit trader": "futures",
    "apex trader funding": "futures",
    "fxify": "cfd",
    "funding predicts": "predictions",
}

_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "how",
    "what", "when", "where", "which", "who", "why", "can", "could", "should",
    "would", "will", "i", "my", "me", "you", "your", "it", "its", "to", "of",
    "in", "on", "for", "and", "or", "with", "at", "by", "about", "get", "have",
    "has", "be", "there", "this", "that", "if", "any", "much", "many", "long",
    "they", "their", "them", "from", "as", "so", "up", "out", "not", "no",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens with stopwords removed."""
    words = re.findall(r"[a-z0-9$%]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


class FAQKnowledgeBase:
    """Loads the bundled FAQ and selects question-relevant excerpts."""

    def __init__(self, data_path: str = DATA_PATH) -> None:
        """Load the knowledge base; missing/broken files degrade to empty.

        Args:
            data_path: Path to the bundled FAQ JSON.
        """
        self.version: str = "none"
        self._sections: Dict[str, Dict[str, object]] = {}
        try:
            with open(data_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.version = str(data.get("version", "unknown"))
            self._sections = data.get("sections", {}) or {}
            total = sum(len(s.get("chunks", [])) for s in self._sections.values())
            logger.info(
                "FAQ knowledge base loaded: %d sections, %d chunks (version %s).",
                len(self._sections), total, self.version,
            )
        except (OSError, ValueError):
            logger.exception("Could not load FAQ knowledge base from %s.", data_path)

    # -- visibility --------------------------------------------------------

    @staticmethod
    def _visible(market: str, markets: Optional[Set[str]], include: Optional[Set[str]], name: str) -> bool:
        """Apply the bot's market visibility rules to a section/entry."""
        if market == "general" or markets is None:
            return True
        if market in markets:
            return True
        return bool(include) and name.strip().lower() in include

    def _hidden_firm_names(self, markets: Optional[Set[str]], include: Optional[Set[str]]) -> List[str]:
        """Firm names that must not surface under the current visibility rules."""
        hidden = []
        for name, market in _KNOWN_FIRM_NAMES.items():
            if not self._visible(market, markets, include, name):
                hidden.append(name)
        return hidden

    # -- retrieval ---------------------------------------------------------

    def mentioned_sections(self, question: str) -> List[str]:
        """Return FAQ section keys for firms explicitly named in the question."""
        low = question.lower()
        found: List[str] = []
        for alias, key in FIRM_ALIASES.items():
            if alias in low and key in self._sections and key not in found:
                found.append(key)
        return found

    def select_excerpts(
        self,
        question: str,
        markets: Optional[Iterable[str]] = None,
        include_names: Optional[Iterable[str]] = None,
        max_chars: int = 20_000,
    ) -> str:
        """Select the FAQ chunks most relevant to a question.

        Args:
            question: The user's question (or a synthetic one, e.g. for compare).
            markets: Allowed markets (``None`` = all visible).
            include_names: Firm names visible regardless of market.
            max_chars: Budget for the returned excerpt text.

        Returns:
            Concatenated excerpt text (may be empty when nothing matches).
        """
        if not self._sections or not question.strip():
            return ""

        markets_set = set(markets) if markets else None
        include_set = (
            {n.strip().lower() for n in include_names} if include_names else None
        )
        hidden = self._hidden_firm_names(markets_set, include_set)

        q_terms = set(_tokenize(question))
        mentioned = set(self.mentioned_sections(question))

        scored: List[Tuple[float, str, str]] = []  # (score, section_key, chunk)
        for key, section in self._sections.items():
            market = str(section.get("market", "general"))
            name = str(section.get("name", key))
            if not self._visible(market, markets_set, include_set, name):
                continue
            boost = 3.0 if key in mentioned else 1.0
            for chunk in section.get("chunks", []):
                low = chunk.lower()
                # Never surface chunks that talk about hidden firms.
                if any(h in low for h in hidden):
                    continue
                c_terms = set(_tokenize(chunk))
                overlap = len(q_terms & c_terms)
                if overlap == 0 and key not in mentioned:
                    continue
                score = overlap * boost
                # Question-style chunks that share terms with the user's
                # question in their first line are usually direct hits.
                first_line = low.split("\n", 1)[0]
                score += sum(1.5 for t in q_terms if t in first_line)
                if score > 0:
                    scored.append((score, key, chunk))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)

        picked: List[str] = []
        used = 0
        seen: Set[int] = set()
        for score, key, chunk in scored:
            if id(chunk) in seen:
                continue
            if used + len(chunk) > max_chars and picked:
                continue
            seen.add(id(chunk))
            picked.append(chunk)
            used += len(chunk)
            if used >= max_chars:
                break

        return "\n\n---\n\n".join(picked)

    def help_center_lines(
        self,
        markets: Optional[Iterable[str]] = None,
        include_names: Optional[Iterable[str]] = None,
    ) -> str:
        """Official help-center links for all visible firms (one per line)."""
        markets_set = set(markets) if markets else None
        include_set = (
            {n.strip().lower() for n in include_names} if include_names else None
        )
        lines = []
        for entry in HELP_CENTERS:
            if self._visible(entry["market"], markets_set, include_set, entry["name"]):
                lines.append(f"- {entry['name']}: {entry['url']}")
        return "\n".join(lines)
