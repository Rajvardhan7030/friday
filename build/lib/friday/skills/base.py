"""Base Skill Protocol for FRIDAY."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class SkillResult(BaseModel):
    """Result returned by a skill execution."""
    success: bool
    data: Any
    message: Optional[str] = None

class BaseSkill(ABC):
    """Base class for all skills."""

    # Security: By default, skills are assumed safe. User skills can be flagged as dangerous.
    __dangerous__: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """The kebab-case name of the skill."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """The description of the skill for tool selection."""
        pass

    @property
    def required_env(self) -> List[str]:
        """List of required environment variables."""
        return []

    @abstractmethod
    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """Execute the skill."""
        pass
