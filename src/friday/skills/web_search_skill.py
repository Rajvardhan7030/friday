"""Async DuckDuckGo HTML search integration skill."""

import asyncio
import logging
import re
import time
from html import unescape
from typing import Any, Dict, List

import httpx

from .base import BaseSkill, SkillResult
from ..core.mcp import MCPToolSchema
from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)

class WebSearchSchema(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")

class WebSearchSkill(BaseSkill):
    """Async web search skill using DuckDuckGo's HTML endpoint."""

    SEARCH_URL = "https://html.duckduckgo.com/html/"
    USER_AGENT = "FRIDAY/0.1 (+https://local-first.assistant)"

    def __init__(self):
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._last_search_time: float = 0.0
        self._rate_limit_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "web-search"

    @property
    def description(self) -> str:
        return "Searches the web for information using DuckDuckGo (no API key required)."

    @property
    def input_schema(self) -> MCPToolSchema:
        return MCPToolSchema.from_model(WebSearchSchema)

    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """Execute web search query with rate limiting and caching."""
        if query in self._cache:
            logger.debug(f"Cache hit for web search: {query}")
            return SkillResult(success=True, data=self._cache[query])

        await self._acquire_search_slot()

        try:
            results = await self._search(query)

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

    async def _search(self, query: str) -> List[Dict[str, Any]]:
        """Fetch and parse DuckDuckGo HTML results asynchronously."""
        html = await self._fetch_search_page(query)
        return self._parse_results(html)

    async def _fetch_search_page(self, query: str) -> str:
        """Call the DuckDuckGo HTML endpoint using an async HTTP client."""
        async with httpx.AsyncClient(
            headers={"User-Agent": self.USER_AGENT},
            follow_redirects=True,
            timeout=10.0,
        ) as client:
            response = await client.get(self.SEARCH_URL, params={"q": query})
            response.raise_for_status()
            return response.text

    @staticmethod
    def _parse_results(html: str) -> List[Dict[str, Any]]:
        """Extract a small structured result set from DuckDuckGo HTML."""
        title_matches = re.findall(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet_matches = re.findall(
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|'
            r'<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        snippets = [
            WebSearchSkill._clean_html(anchor_snippet or div_snippet)
            for anchor_snippet, div_snippet in snippet_matches
        ]

        results: List[Dict[str, Any]] = []
        for index, (href, raw_title) in enumerate(title_matches[:5]):
            results.append(
                {
                    "title": WebSearchSkill._clean_html(raw_title),
                    "href": unescape(href),
                    "body": snippets[index] if index < len(snippets) else "",
                }
            )
        return results

    @staticmethod
    def _clean_html(value: str) -> str:
        """Collapse simple HTML fragments into plain text."""
        value = re.sub(r"<[^>]+>", "", value)
        value = unescape(value)
        return " ".join(value.split())
