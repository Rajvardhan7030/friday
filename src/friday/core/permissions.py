"""Centralized Permission Management for Friday."""

import logging
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel
from .config import Config

logger = logging.getLogger(__name__)

class ToolPermission(BaseModel):
    """Definition of a permission required by a tool."""
    name: str
    description: str
    level: str = "medium" # low, medium, high, critical

class PermissionManager:
    """Manages tool execution permissions and user confirmations."""

    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self._trusted_tools: Set[str] = set(config.get("security.trusted_tools", []))
        self._blocked_tools: Set[str] = set(config.get("security.blocked_tools", []))
        self._auto_confirm_level: str = config.get("security.auto_confirm_level", "low")

    async def check_permission(self, tool_name: str, args: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Verify if a tool has permission to execute."""
        if tool_name in self._blocked_tools:
            logger.warning(f"Blocked tool execution attempt: {tool_name}")
            return False

        if tool_name in self._trusted_tools:
            return True

        # Determine level (default to medium)
        level = "medium"
        if metadata:
            level = metadata.get("level", "medium")

        # Handle auto-confirmation based on level
        if self._should_auto_confirm(level):
            return True

        # Explicit confirmation
        return await self._ask_user_confirmation(tool_name, args, level)

    def _should_auto_confirm(self, level: str) -> bool:
        """Check if an action level should be automatically confirmed."""
        levels = ["low", "medium", "high", "critical"]
        try:
            auto_idx = levels.index(self._auto_confirm_level)
            lvl_idx = levels.index(level)
            return lvl_idx <= auto_idx
        except ValueError:
            return False

    async def _ask_user_confirmation(self, tool_name: str, args: Dict[str, Any], level: str) -> bool:
        """Prompt the user to allow a specific tool execution."""
        # In a real CLI, we wait for input. 
        # For autonomous agents, this is where we'd need a pause/resume or a UI notification.
        
        self.console.print(Panel(
            f"Friday wants to use tool: [bold cyan]{tool_name}[/bold cyan]\n"
            f"Level: [bold yellow]{level.upper()}[/bold yellow]\n"
            f"Arguments: {args}",
            title="Permission Request",
            border_style="yellow"
        ))
        
        # Check config for non-interactive mode
        if self.config.get("security.non_interactive", False):
            logger.warning(f"Rejecting tool {tool_name} in non-interactive mode.")
            return False

        confirmed = Confirm.ask(f"Allow [cyan]{tool_name}[/cyan] to proceed?")
        
        if confirmed:
            if Confirm.ask(f"Always trust [cyan]{tool_name}[/cyan]?"):
                self._trusted_tools.add(tool_name)
                # We should ideally save this to config
                current_trusted = self.config.get("security.trusted_tools", [])
                if tool_name not in current_trusted:
                    current_trusted.append(tool_name)
                    self.config.set("security.trusted_tools", current_trusted)
                    self.config.save()
        
        return confirmed
