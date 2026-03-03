"""
DuckDuckGo web search with offline fallback and rate limiting.
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum seconds between searches to avoid rate-limiting
_RATE_LIMIT_SECONDS = 10


class WebSearcher:
    """Wraps the duckduckgo-search library with rate limiting and error handling."""

    def __init__(self, config: Dict) -> None:
        """Initialise the searcher.

        Args:
            config: Full parsed config dictionary.
        """
        search_cfg = config.get("search", {})
        self.enabled: bool = search_cfg.get("enabled", True)
        self.max_results: int = search_cfg.get("max_results", 3)
        self._last_search_time: float = 0.0

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """Perform a DuckDuckGo search and return structured results.

        Respects the 10-second rate limit between searches.  Falls back
        to an empty list on network errors or when search is disabled.

        Args:
            query: The search query string.
            max_results: Override for the configured max_results value.

        Returns:
            List of dicts with keys ``title``, ``url``, ``snippet``.
        """
        if not self.enabled:
            logger.debug("Web search is disabled in config.")
            return []

        if not query.strip():
            return []

        n = max_results if max_results is not None else self.max_results

        # Enforce rate limit
        elapsed = time.monotonic() - self._last_search_time
        if elapsed < _RATE_LIMIT_SECONDS:
            wait = _RATE_LIMIT_SECONDS - elapsed
            logger.debug("Rate limiting: sleeping %.1f s before search.", wait)
            time.sleep(wait)

        self._last_search_time = time.monotonic()

        try:
            from duckduckgo_search import DDGS

            results: List[Dict] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=n):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        }
                    )
            logger.info("Search for '%s' returned %d results.", query, len(results))
            return results
        except Exception as exc:
            # HTTP 202 or other transient errors — return empty list gracefully
            logger.warning("Web search failed for query '%s': %s", query, exc)
            return []
