"""Browser control skill for FRIDAY via a Go-based daemon."""

import logging
import httpx
from typing import Dict, Any, List, Optional
from pydantic import Field

from .base import BaseSkill, SkillResult
from ..core.mcp import MCPToolSchema

logger = logging.getLogger(__name__)

class BrowserNavigateSchema(MCPToolSchema):
    url: str = Field(..., description="The URL to navigate to")
    profile: str = Field("default", description="The browser profile to use")
    headless: bool = Field(True, description="Whether to run in headless mode")

class BrowserActionSchema(MCPToolSchema):
    type: str = Field(..., description="The action type: 'click' or 'type'")
    selector: str = Field(..., description="The CSS selector for the target element")
    value: str = Field("", description="The value to type (if applicable)")
    profile: str = Field("default", description="The browser profile to use")

class BrowserSkill(BaseSkill):
    """Skill to control a web browser via a local Go daemon."""

    def __init__(self, daemon_url: str = "http://localhost:9000"):
        self.daemon_url = daemon_url

    @property
    def name(self) -> str:
        return "browser-control"

    @property
    def description(self) -> str:
        return (
            "Controls a web browser to navigate sites, extract text, and perform actions. "
            "Use this for searching information, logging into sites, and downloading reports."
        )

    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """
        Execute a browser command. 
        'query' is unused here as we prefer structured tool calls via MCP.
        """
        return SkillResult(success=False, data=None, message="Use structured tool calls for browser control.")

    async def navigate(self, url: str, profile: str = "default", headless: bool = True) -> SkillResult:
        """Navigate to a URL and return extracted text."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.daemon_url}/navigate",
                    json={"url": url, "profile": profile, "headless": headless}
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("success"):
                    return SkillResult(success=True, data=data.get("content"))
                return SkillResult(success=False, data=None, message=data.get("message"))
        except Exception as e:
            logger.error(f"Browser navigation failed: {e}")
            return SkillResult(success=False, data=None, message=str(e))

    async def perform_action(self, action_type: str, selector: str, value: str = "", profile: str = "default") -> SkillResult:
        """Perform an action (click, type) on the current page."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.daemon_url}/action",
                    json={"type": action_type, "selector": selector, "value": value, "profile": profile}
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("success"):
                    return SkillResult(success=True, data="Action performed successfully")
                return SkillResult(success=False, data=None, message=data.get("message"))
        except Exception as e:
            logger.error(f"Browser action failed: {e}")
            return SkillResult(success=False, data=None, message=str(e))

    # Overriding register_mcp to register multiple tools for this skill
    def register_mcp(self):
        """Registers the specific browser tools to the global MCP client."""
        from ..core.mcp import mcp_client, MCPTool
        
        # 1. Navigate Tool
        mcp_client.register_tool(
            MCPTool(
                name="browser_navigate",
                description="Navigate to a URL and extract text content.",
                inputSchema=BrowserNavigateSchema()
            ),
            self._mcp_navigate_handler
        )
        
        # 2. Action Tool
        mcp_client.register_tool(
            MCPTool(
                name="browser_action",
                description="Perform an action (click, type) on the current web page.",
                inputSchema=BrowserActionSchema()
            ),
            self._mcp_action_handler
        )

    async def _mcp_navigate_handler(self, **kwargs) -> Any:
        url = kwargs.get("url")
        profile = kwargs.get("profile", "default")
        headless = kwargs.get("headless", True)
        result = await self.navigate(url, profile, headless)
        if result.success:
            return result.data
        return f"Error: {result.message}"

    async def _mcp_action_handler(self, **kwargs) -> Any:
        action_type = kwargs.get("type")
        selector = kwargs.get("selector")
        value = kwargs.get("value", "")
        profile = kwargs.get("profile", "default")
        result = await self.perform_action(action_type, selector, value, profile)
        if result.success:
            return result.data
        return f"Error: {result.message}"
