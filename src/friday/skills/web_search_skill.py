"""DuckDuckGo local search integration skill."""

import logging
from typing import Dict, Any, List
from duckduckgo_search import DDGS
from src.friday.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

class WebSearchSkill(BaseSkill):
    """Local web search skill using DuckDuckGo."""

    @property
    def name(self) -> str:
        return "web-search"

    @property
    def description(self) -> str:
        return "Searches the web for information using DuckDuckGo (no API key required)."

    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """Execute web search query."""
        try:
            results = []
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(query, max_results=5)
                for r in ddgs_gen:
                    results.append(r)

            if not results:
                return SkillResult(success=False, data=[], message="No results found.")

            return SkillResult(success=True, data=results)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return SkillResult(success=False, data=[], message=str(e))
