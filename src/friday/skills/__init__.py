"""Friday Skills."""

from .base import BaseSkill, SkillResult
from .web_search_skill import WebSearchSkill
from .browser_skill import BrowserSkill

__all__ = ["BaseSkill", "SkillResult", "WebSearchSkill", "BrowserSkill"]
