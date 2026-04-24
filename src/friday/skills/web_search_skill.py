"""DuckDuckGo local search integration skill."""

import logging
import asyncio
import time
from typing import Dict, Any
from .base import BaseSkill, SkillResult

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)

class WebSearchSkill(BaseSkill):
    """Local web search skill using DuckDuckGo with rate limiting and caching."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._last_search_time: float = 0.0

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

        # 2. Rate limit: wait at least 1s between searches
        now = time.time()
        elapsed = now - self._last_search_time
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)

        try:
            results = []
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(query, max_results=5)
                if ddgs_gen:
                    for r in ddgs_gen:
                        results.append(r)

            self._last_search_time = time.time()

            if not results:
                return SkillResult(success=False, data=[], message="No results found.")

            # Store in cache
            self._cache[query] = results
            return SkillResult(success=True, data=results)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return SkillResult(success=False, data=[], message=str(e))
