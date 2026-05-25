"""Self-Healing and Recovery for Friday workflows."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Awaitable
from .exceptions import FridayError, ModelNotFoundError, BrowserDaemonUnavailable, LLMError
from rich.console import Console
from rich.prompt import Confirm

logger = logging.getLogger(__name__)

class RecoveryStrategy:
    """Definition of a strategy to recover from a specific error."""
    def __init__(self, error_type: type, description: str, handler: Callable[..., Awaitable[bool]]):
        self.error_type = error_type
        self.description = description
        self.handler = handler

class RecoveryManager:
    """Orchestrates error recovery and user remediation."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._strategies: Dict[type, RecoveryStrategy] = {}
        self._register_default_strategies()

    def register_strategy(self, strategy: RecoveryStrategy):
        """Register a new recovery strategy."""
        self._strategies[strategy.error_type] = strategy

    async def attempt_recovery(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """Find and execute a strategy to recover from the given error."""
        error_type = type(error)
        strategy = self._strategies.get(error_type)
        
        if not strategy:
            # Check for base classes
            for st_type, st in self._strategies.items():
                if isinstance(error, st_type):
                    strategy = st
                    break
        
        if not strategy:
            return False

        logger.info(f"Attempting recovery for {error_type.__name__}: {strategy.description}")
        return await strategy.handler(error, context or {})

    def _register_default_strategies(self):
        """Register built-in recovery logic."""
        
        # 1. Model Missing Recovery
        async def recover_missing_model(error: ModelNotFoundError, ctx: Dict[str, Any]) -> bool:
            model_name = getattr(error, "model_name", "unknown")
            self.console.print(f"[bold yellow]Heal:[/bold yellow] Model '{model_name}' is missing.")
            if Confirm.ask(f"Would you like Friday to attempt to download it now?"):
                from ..cli import ollama_pull
                return await ollama_pull(model_name)
            return False

        self.register_strategy(RecoveryStrategy(
            ModelNotFoundError, 
            "Download missing LLM or voice models", 
            recover_missing_model
        ))

        # 2. Browser Daemon Offline Recovery
        async def recover_browser_daemon(error: Exception, ctx: Dict[str, Any]) -> bool:
            self.console.print("[bold yellow]Heal:[/bold yellow] Browser automation daemon is offline.")
            self.console.print("Try running: [cyan]cd src/friday/skills/browser_daemon && go run main.go[/cyan]")
            return False # Manual fix required for now

        self.register_strategy(RecoveryStrategy(
            BrowserDaemonUnavailable,
            "Suggest commands to start browser daemon",
            recover_browser_daemon
        ))

# Global instance
recovery_manager = RecoveryManager()
