"""Model Context Protocol (MCP) definitions and client interface."""

import logging
from typing import Dict, Any, List, Callable, Awaitable, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MCPToolSchema(BaseModel):
    """JSON Schema for the tool's input arguments."""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)

class MCPTool(BaseModel):
    """Metadata for an MCP Tool."""
    name: str
    description: str
    inputSchema: MCPToolSchema

class MCPClient:
    """A client to manage and execute MCP tools."""
    def __init__(self):
        self._tools: Dict[str, tuple[MCPTool, Callable[..., Awaitable[Any]]]] = {}

    def register_tool(self, tool: MCPTool, handler: Callable[..., Awaitable[Any]]):
        """Registers a tool with the MCP client."""
        self._tools[tool.name] = (tool, handler)
        logger.info(f"Registered MCP tool: {tool.name}")

    def list_tools(self) -> List[MCPTool]:
        """Returns a list of registered tools."""
        return [t[0] for t in self._tools.values()]

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Returns tools in the format expected by the LLM (OpenAI/Ollama format)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema.model_dump()
                }
            }
            for tool, _ in self._tools.values()
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Calls a tool by name with the given arguments."""
        if name not in self._tools:
            raise ValueError(f"Unknown MCP tool: {name}")
        _, handler = self._tools[name]
        try:
            return await handler(**arguments)
        except Exception as e:
            logger.error(f"Error executing MCP tool {name}: {e}")
            raise

# Global MCP client instance
mcp_client = MCPClient()
