"""Browser control skill for FRIDAY via a Go-based daemon."""

import logging
import httpx
from typing import Dict, Any, List, Optional
from pydantic import Field, BaseModel

from .base import BaseSkill, SkillResult
from ..core.mcp import MCPToolSchema
from ..core.exceptions import BrowserDaemonUnavailable

logger = logging.getLogger(__name__)

class BrowserNavigateSchema(BaseModel):
    url: str = Field(..., description="The URL to navigate to")
    profile: str = Field("default", description="The browser profile to use")
    headless: bool = Field(True, description="Whether to run in headless mode")

class BrowserActionSchema(BaseModel):
    type: str = Field(..., description="The action type: 'click', 'type', or 'content'")
    selector: str = Field(..., description="The CSS selector for the target element")
    value: str = Field("", description="The value to type (if applicable)")
    profile: str = Field("default", description="The browser profile to use")
    page_id: Optional[str] = Field(None, description="The ID of the page to act on")

class BrowserPagesSchema(BaseModel):
    profile: str = Field("default", description="The browser profile to use")

class BrowserCloseSchema(BaseModel):
    page_id: str = Field(..., description="The ID of the page to close")
    profile: str = Field("default", description="The browser profile to use")

class BrowserSkill(BaseSkill):
    """Skill to control a web browser via a local Go daemon."""

    def __init__(self, daemon_url: str = "http://localhost:9000"):
        self.daemon_url = daemon_url

    async def is_daemon_alive(self) -> bool:
        """Check if the browser daemon is reachable."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.daemon_url}/health")
                # The daemon process is running if it responds to HTTP requests,
                # even if it returns 503 (e.g. missing browser).
                return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return "browser-control"

    @property
    def description(self) -> str:
        return (
            "Controls a web browser to navigate sites, extract text, and perform actions. "
            "Supports multiple persistent pages and sessions."
        )

    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """
        Execute a browser command. 
        'query' is unused here as we prefer structured tool calls via MCP.
        """
        return SkillResult(success=False, data=None, message="Use structured tool calls for browser control.")

    async def navigate(self, url: str, profile: str = "default", headless: bool = True) -> SkillResult:
        """Navigate to a URL and return extracted text and page_id."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.daemon_url}/navigate",
                    json={"url": url, "profile": profile, "headless": headless}
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("success"):
                    return SkillResult(
                        success=True, 
                        data={
                            "content": data.get("content"),
                            "page_id": data.get("page_id")
                        }
                    )
                return SkillResult(success=False, data=None, message=data.get("message"))
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise BrowserDaemonUnavailable(f"Browser daemon at {self.daemon_url} is unreachable: {e}")
        except Exception as e:
            logger.error(f"Browser navigation failed: {e}")
            return SkillResult(success=False, data=None, message=str(e))

    async def perform_action(self, action_type: str, selector: str, value: str = "", profile: str = "default", page_id: Optional[str] = None) -> SkillResult:
        """Perform an action (click, type) on a specific page."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {"type": action_type, "selector": selector, "value": value, "profile": profile}
                if page_id:
                    payload["page_id"] = page_id
                    
                resp = await client.post(f"{self.daemon_url}/action", json=payload)
                resp.raise_for_status()
                data = resp.json()
                if data.get("success"):
                    return SkillResult(success=True, data=data.get("content") or "Action performed successfully")
                return SkillResult(success=False, data=None, message=data.get("message"))
        except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise BrowserDaemonUnavailable(f"Browser daemon at {self.daemon_url} is unreachable: {e}")
        except Exception as e:
            logger.error(f"Browser action failed: {e}")
            return SkillResult(success=False, data=None, message=str(e))

    async def list_pages(self, profile: str = "default") -> SkillResult:
        """List active pages."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.daemon_url}/pages", params={"profile": profile})
                resp.raise_for_status()
                data = resp.json()
                if data.get("success"):
                    return SkillResult(success=True, data=data.get("pages"))
                return SkillResult(success=False, message=data.get("message"))
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise BrowserDaemonUnavailable(f"Browser daemon at {self.daemon_url} is unreachable: {e}")
        except Exception as e:
            return SkillResult(success=False, message=str(e))

    async def close_page(self, page_id: str, profile: str = "default") -> SkillResult:
        """Close a specific page."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self.daemon_url}/close", json={"page_id": page_id, "profile": profile})
                resp.raise_for_status()
                data = resp.json()
                return SkillResult(success=data.get("success"), message=data.get("message"))
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise BrowserDaemonUnavailable(f"Browser daemon at {self.daemon_url} is unreachable: {e}")
        except Exception as e:
            return SkillResult(success=False, message=str(e))

    # Overriding register_mcp to register multiple tools for this skill
    def register_mcp(self):
        """Registers the specific browser tools to the global MCP client."""
        from ..core.mcp import mcp_client, MCPTool
        
        # 1. Navigate Tool
        mcp_client.register_tool(
            MCPTool(
                name="browser_navigate",
                description="Open a URL and return text content and a page_id for future actions.",
                inputSchema=MCPToolSchema.from_model(BrowserNavigateSchema)
            ),
            self._mcp_navigate_handler
        )
        
        # 2. Action Tool
        mcp_client.register_tool(
            MCPTool(
                name="browser_action",
                description="Perform an action (click, type, content) on a web page. Use page_id if available.",
                inputSchema=MCPToolSchema.from_model(BrowserActionSchema)
            ),
            self._mcp_action_handler
        )
        
        # 3. List Pages
        mcp_client.register_tool(
            MCPTool(
                name="browser_list_pages",
                description="List all active persistent browser pages.",
                inputSchema=MCPToolSchema.from_model(BrowserPagesSchema)
            ),
            self._mcp_list_handler
        )
        
        # 4. Close Page
        mcp_client.register_tool(
            MCPTool(
                name="browser_close_page",
                description="Close a persistent browser page by its ID.",
                inputSchema=MCPToolSchema.from_model(BrowserCloseSchema)
            ),
            self._mcp_close_handler
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
        page_id = kwargs.get("page_id")
        result = await self.perform_action(action_type, selector, value, profile, page_id)
        if result.success:
            return result.data
        return f"Error: {result.message}"

    async def _mcp_list_handler(self, **kwargs) -> Any:
        profile = kwargs.get("profile", "default")
        result = await self.list_pages(profile)
        if result.success:
            return result.data
        return f"Error: {result.message}"

    async def _mcp_close_handler(self, **kwargs) -> Any:
        page_id = kwargs.get("page_id")
        profile = kwargs.get("profile", "default")
        result = await self.close_page(page_id, profile)
        if result.success:
            return "Page closed successfully"
        return f"Error: {result.message}"
