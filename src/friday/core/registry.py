"""Command registry for Friday agents and features."""

import re
import logging
from typing import Dict, List, Callable, Optional, Pattern, Any, Awaitable, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# To maintain compatibility, we'll auto-register legacy commands as plugins
from .plugin import plugin_manager, PluginManifest

@dataclass
class Command:
    """Metadata for a registered command."""
    name: str
    pattern: Pattern
    handler: Callable[..., Awaitable[Any]]
    description: str
    help_usage: str
    priority: int = 0

class CommandRegistry:
    """Central registry to map user intent to specific handlers."""

    def __init__(self):
        self._commands: List[Command] = []

    def register(self, name: str, regex: str, description: str, usage: str, priority: int = 0):
        """Decorator to register a new command."""
        def decorator(func: Callable[..., Awaitable[Any]]):
            compiled_pattern = re.compile(regex, re.IGNORECASE)
            command = Command(name, compiled_pattern, func, description, usage, priority)
            self._commands.append(command)
            # Sort by priority (higher first)
            self._commands.sort(key=lambda x: x.priority, reverse=True)
            
            # Compatibility wrapper: auto-register as a legacy plugin if not already tracked
            plugin_name = f"legacy_command_{name.lower().replace(' ', '_')}"
            if plugin_name not in plugin_manager.plugins:
                manifest = PluginManifest(
                    name=plugin_name,
                    version="0.1.0",
                    description=f"Legacy command wrapper for {name}",
                    author="Friday Core",
                    entry_point=func.__module__
                )
                plugin_manager.plugins[plugin_name] = manifest

            return func
        return decorator

    def find_handler(self, text: str) -> Optional[Tuple[Command, re.Match]]:
        """Find the first command that matches the input text."""
        for cmd in self._commands:
            match = cmd.pattern.search(text)
            if match:
                return cmd, match
        return None

    def get_help(self) -> str:
        """Generate a formatted help string for all commands."""
        help_text = "[bold blue]Available Commands:[/bold blue]\n"
        # Group by name to avoid duplicates if any, though name should be unique
        for cmd in sorted(self._commands, key=lambda x: x.name):
            help_text += f"• [bold cyan]{cmd.name:15}[/bold cyan] {cmd.description}\n"
            help_text += f"  [dim]Usage: {cmd.help_usage}[/dim]\n"
        return help_text

# Global registry instance
registry = CommandRegistry()
