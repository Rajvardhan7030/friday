"""RSS/Atom news aggregator skill."""

import logging
import httpx
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
from src.friday.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

class NewsSkill(BaseSkill):
    """News aggregator skill for RSS feeds."""

    @property
    def name(self) -> str:
        return "news"

    @property
    def description(self) -> str:
        return "Gathers top news headlines from configurable RSS feeds."

    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """Fetch news headlines from feeds."""
        # For v0.1 we'll use some default feeds, but these should come from config
        feeds = [
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
        ]
        
        all_news = []
        async with httpx.AsyncClient() as client:
            for url in feeds:
                try:
                    response = await client.get(url, timeout=10.0)
                    if response.status_code == 200:
                        headlines = self._parse_rss(response.text)
                        all_news.extend(headlines[:5]) # top 5 per feed
                except Exception as e:
                    logger.warning(f"Failed to fetch news from {url}: {e}")

        if not all_news:
            return SkillResult(success=False, data=[], message="No news found.")

        return SkillResult(success=True, data=all_news)

    def _parse_rss(self, xml_content: str) -> List[Dict[str, str]]:
        """Parse RSS XML content."""
        headlines = []
        try:
            root = ET.fromstring(xml_content)
            for item in root.findall(".//item"):
                title = item.find("title")
                link = item.find("link")
                if title is not None:
                    headlines.append({
                        "title": title.text,
                        "link": link.text if link is not None else ""
                    })
        except Exception as e:
            logger.error(f"Failed to parse news RSS: {e}")
        return headlines
