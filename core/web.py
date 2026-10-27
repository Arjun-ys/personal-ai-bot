import urllib.request
from html.parser import HTMLParser

from core.config import (
    _PERSONAL_QUERY_KEYWORDS,
    _EXPLICIT_WEB_PHRASES,
    _WEB_QUERY_PATTERNS,
    _WEB_FOLLOWUP_PATTERN,
    _WEB_FOLLOWUP_NUMBER,
    _ORDINAL_MAP,
    _DATA_KEYWORDS,
    _CASUAL_PATTERNS,
)

# DuckDuckGo Search (free web access — no API key needed)
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

URL_FETCH_AVAILABLE = True


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML→text converter for URL fetching."""

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self._pieces.append(text)

    def get_text(self) -> str:
        return " ".join(self._pieces)


def _fetch_url_text(url: str, max_chars: int = 8000) -> str:
    """Download a URL and return cleaned plain text."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (PersonalAI Bot)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        extractor = _HTMLTextExtractor()
        extractor.feed(raw)
        text = extractor.get_text()
        return text[:max_chars] if len(text) > max_chars else text
    except Exception as e:
        print(f"[URL Fetch Error] {e}")
        return ""


def search_web_ddg(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo text search and return formatted results.
    Also stores raw results in session state for follow-up queries."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        # Store raw results for follow-up references ("go over the 2nd result")
        try:
            import streamlit as st
            st.session_state["_last_web_results"] = results
        except Exception:
            pass

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"[{i}] {title}\n    {body}\n    Source: {href}")
        return "\n\n".join(lines)
    except Exception as e:
        print(f"[Web Search Error] {e}")
        return ""


_search_web = search_web_ddg


def _is_personal_query(query: str) -> bool:
    """Check if query is related to user's personal vault data."""
    q_lower = query.lower().strip()
    for kw in _PERSONAL_QUERY_KEYWORDS:
        if kw in q_lower:
            return True
    return False


def _should_auto_web_search(query: str) -> bool:
    """Decide whether a user query should trigger a web search."""
    q_lower = query.lower().strip()
    # Block personal / vault queries from going to web
    if _is_personal_query(query):
        return False
    # Explicit web-intent phrases → always trigger
    for phrase in _EXPLICIT_WEB_PHRASES:
        if phrase in q_lower:
            return True
    # Match a web-style pattern
    return bool(_WEB_QUERY_PATTERNS.search(q_lower))


_is_web_query = _should_auto_web_search


def _is_casual_message(user_input: str) -> bool:
    """Detect casual/short messages — don't pollute with irrelevant context."""
    _lower = user_input.lower()
    return (
        len(user_input.split()) <= 3
        and not any(kw in _lower for kw in _DATA_KEYWORDS)
        and not any(p in _lower for p in _CASUAL_PATTERNS)
    )


def _check_web_followup(query: str, last_web_results=None) -> str | None:
    """If the user references a previous web result by number, fetch its full content.
    Returns formatted web_context string, or None if not a follow-up."""
    prev = last_web_results
    if prev is None:
        try:
            import streamlit as st
            prev = st.session_state.get("_last_web_results")
        except Exception:
            prev = None

    if not prev:
        return None

    match = _WEB_FOLLOWUP_PATTERN.search(query)
    if not match:
        return None

    # Extract the ordinal/number
    num_match = _WEB_FOLLOWUP_NUMBER.findall(query)
    if not num_match:
        return None

    # Take the last number found (most likely the intended one)
    raw = num_match[-1].strip().lstrip("#").strip().lower()
    idx = _ORDINAL_MAP.get(raw)
    if idx is None:
        return None

    if idx < 1 or idx > len(prev):
        return None

    result = prev[idx - 1]
    title = result.get("title", "")
    href = result.get("href", "")
    body = result.get("body", "")

    # Fetch the full article
    full_text = ""
    if href:
        full_text = _fetch_url_text(href, max_chars=10000)

    if full_text:
        return (
            f"[Full article from web result #{idx}]\n"
            f"Title: {title}\n"
            f"URL: {href}\n\n"
            f"{full_text}"
        )
    else:
        # Fallback to the snippet
        return (
            f"[Web result #{idx} — could not fetch full page]\n"
            f"Title: {title}\n"
            f"URL: {href}\n"
            f"Summary: {body}"
        )

