"""DuckDuckGo web search with offline fallback and rate limiting.

Performs up to one search per 10 seconds to avoid being throttled.
Returns a list of result dictionaries compatible with the duckduckgo-search
library's output format.
"""

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MIN_SEARCH_INTERVAL = 10  # seconds between searches


class WebSearch:
    """Thin wrapper around ``duckduckgo_search`` with rate limiting.

    Args:
        config: Parsed configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        search_cfg = config.get("search", {})
        self.enabled: bool = search_cfg.get("enabled", True)
        self.max_results: int = search_cfg.get("max_results", 3)
        self._last_search_time: float = 0.0

    def search(self, query: str, max_results: int = 0) -> List[Dict[str, Any]]:
        """Search DuckDuckGo and return results.

        Enforces a minimum interval between searches.  Returns an empty list
        if search is disabled, the query is blank, or an error occurs.

        Args:
            query: Search query string.
            max_results: Override for the configured ``max_results`` value.
                         0 means use the configured default.

        Returns:
            List of result dicts, each containing at least ``title``, ``href``
            (or ``url``), and ``body`` keys.
        """
        if not self.enabled:
            logger.debug("Web search is disabled in config.")
            return []

        if not query.strip():
            return []

        # Rate limiting
        elapsed = time.monotonic() - self._last_search_time
        if elapsed < _MIN_SEARCH_INTERVAL:
            wait = _MIN_SEARCH_INTERVAL - elapsed
            logger.debug("Rate limiting search – waiting %.1f s.", wait)
            time.sleep(wait)

        n = max_results if max_results > 0 else self.max_results

        try:
            from duckduckgo_search import DDGS  # noqa: PLC0415

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=n))
            self._last_search_time = time.monotonic()
            logger.info("Search '%s' returned %d results.", query, len(results))
            return results
        except ImportError:
            logger.warning("duckduckgo-search not installed; returning empty results.")
            return []
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Search failed for '%s': %s", query, exc)
            return []
