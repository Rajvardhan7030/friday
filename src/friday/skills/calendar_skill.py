"""Mock calendar connector skill."""

import logging
from typing import Dict, Any, List
from src.friday.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

class CalendarSkill(BaseSkill):
    """Mock calendar skill for v0.1 demonstration."""

    @property
    def name(self) -> str:
        return "calendar"

    @property
    def description(self) -> str:
        return "Gathers today's calendar events (currently in mock mode)."

    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """Fetch mock today's events."""
        mock_events = [
            {"time": "09:00 AM", "event": "Stand-up meeting"},
            {"time": "12:30 PM", "event": "Lunch with the team"},
            {"time": "03:00 PM", "event": "Architecture review"}
        ]
        
        return SkillResult(
            success=True, 
            data=mock_events,
            message="Mock mode: Found 3 events for today."
        )
