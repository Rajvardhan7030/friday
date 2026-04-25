"""Base Skill Protocol for FRIDAY."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from friday.core.mcp import MCPTool, MCPToolSchema, mcp_client

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
    def input_schema(self) -> MCPToolSchema:
        """Default empty schema if the skill takes no specific arguments."""
        return MCPToolSchema()

    @property
    def required_env(self) -> List[str]:
        """List of required environment variables."""
        return []

    @abstractmethod
    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """Execute the skill."""
        pass

    def as_mcp_tool(self) -> MCPTool:
        """Returns the MCPTool definition for this skill."""
        return MCPTool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema
        )

    def register_mcp(self):
        """Registers the skill to the global MCP client."""
        mcp_client.register_tool(self.as_mcp_tool(), self._mcp_handler)

    async def _mcp_handler(self, **kwargs) -> Any:
        """Adapter handler for MCP calls."""
        # For simplicity, pass all kwargs as query or context
        query = kwargs.get("query", "")
        result = await self.execute(query=query, context=kwargs)
        if result.success:
            return result.data
        return f"Error: {result.message}"
