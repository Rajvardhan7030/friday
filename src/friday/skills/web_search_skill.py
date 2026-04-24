"""DuckDuckGo local search integration skill."""

import logging
import asyncio
import time
import warnings
from typing import Dict, Any
from .base import BaseSkill, SkillResult

try:
    from ddgs import DDGS
    USING_LEGACY_DDGS = False
except ImportError:
    try:
        from duckduckgo_search import DDGS
        USING_LEGACY_DDGS = True
    except ImportError:
        DDGS = None
        USING_LEGACY_DDGS = False

logger = logging.getLogger(__name__)

class WebSearchSkill(BaseSkill):
    """Local web search skill using DuckDuckGo with rate limiting and caching."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._last_search_time: float = 0.0
        self._rate_limit_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "web-search"

    @property
    def description(self) -> str:
        return "Searches the web for information using DuckDuckGo (no API key required)."

    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """Execute web search query with rate limiting and caching."""
        if DDGS is None:
            return SkillResult(success=False, data=[], message="The 'duckduckgo-search' package is not installed.")

        # 1. Check cache first
        if query in self._cache:
            logger.debug(f"Cache hit for web search: {query}")
            return SkillResult(success=True, data=self._cache[query])

        await self._acquire_search_slot()

        try:
            results = await asyncio.to_thread(self._search, query)

            if not results:
                return SkillResult(success=False, data=[], message="No results found.")

            # Store in cache
            self._cache[query] = results
            return SkillResult(success=True, data=results)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return SkillResult(success=False, data=[], message=str(e))

    async def _acquire_search_slot(self) -> None:
        """Reserve the next allowed search slot to enforce global spacing."""
        async with self._rate_limit_lock:
            now = time.monotonic()
            elapsed = now - self._last_search_time
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
                now = time.monotonic()

            self._last_search_time = now

    @staticmethod
    def _search(query: str) -> list[Dict[str, Any]]:
        """Run the synchronous DDGS search call off the event loop."""
        results = []
        if USING_LEGACY_DDGS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with DDGS() as ddgs:
                    ddgs_gen = ddgs.text(query, max_results=5)
                    if ddgs_gen:
                        for result in ddgs_gen:
                            results.append(result)
            return results

        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(query, max_results=5)
            if ddgs_gen:
                for result in ddgs_gen:
                    results.append(result)
        return results
