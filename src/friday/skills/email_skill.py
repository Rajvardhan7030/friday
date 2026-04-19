"""Mock email connector skill."""

import logging
from typing import Dict, Any, List
from src.friday.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

class EmailSkill(BaseSkill):
    """Mock email skill for v0.1 demonstration."""

    @property
    def name(self) -> str:
        return "email"

    @property
    def description(self) -> str:
        return "Gathers unread emails from the last 24h (currently in mock mode)."

    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """Gathers mock unread emails."""
        # v0.1 will use mock data unless real creds are provided
        mock_emails = [
            {"from": "boss@example.com", "subject": "Project Deadline", "snippet": "Hey, how is the AI assistant going?"},
            {"from": "newsletter@tech.com", "subject": "Daily Tech Brief", "snippet": "New breakthroughs in local LLMs..."},
            {"from": "mom@home.com", "subject": "Dinner Sunday?", "snippet": "Are you coming over this weekend?"}
        ]
        
        return SkillResult(
            success=True, 
            data=mock_emails,
            message="Mock mode: Found 3 unread emails."
        )
