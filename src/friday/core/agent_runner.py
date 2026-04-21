"""Orchestrator for managing sessions and executing commands."""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from .registry import registry
from .config import Config
from .exceptions import FridayError, ModelNotFoundError, SandboxError
from ..llm.local import LocalEngine
from ..llm.engine import Message

logger = logging.getLogger(__name__)

class Session:
# ... (rest of Session class stays same)
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.last_action: Optional[str] = None
        self.variables: Dict[str, Any] = {}

    def add_message(self, role: str, content: str):
        """Add a message to the session history."""
        self.history.append({"role": role, "content": content})
        # Keep history manageable (e.g., last 20 messages)
        if len(self.history) > 20:
            self.history.pop(0)

class AgentRunner:
    """The 'Brain' that decides how to handle an input."""

    def __init__(self, config: Config):
        self.config = config
        self.session = Session()
        
        # Initialize LLM Engine
        self.llm = LocalEngine(
            primary_model=config.get("llm.primary_model"),
            fallback_model=config.get("llm.fallback_model"),
            base_url=config.get("llm.base_url")
        )
        
        self._load_agents()

    def _load_agents(self):
        """Import agent modules to trigger registry decorators."""
        try:
            from ..agents import system_commands
            from ..agents import code_assistant
            # The code_assistant will register itself for task-based intents
        except ImportError as e:
            logger.warning(f"Could not load some agents: {e}")

    async def handle_input(self, text: str) -> str:
        """Main entry point for processing any user input."""
        text = text.strip()
        if not text:
            return "I'm listening, but I didn't hear anything."

        # 1. Check the Command Registry (Deterministic Logic)
        handler_data = registry.find_handler(text)
        if handler_data:
            cmd, match = handler_data
            logger.info(f"Executing command: {cmd.name}")
            try:
                # Pass session and regex matches to the handler
                # Some handlers might need self.llm or self.config
                result = await cmd.handler(self.session, *match.groups(), llm=self.llm, config=self.config)
                self.session.add_message("user", text)
                self.session.add_message("assistant", str(result))
                return result
            except Exception as e:
                logger.error(f"Command {cmd.name} failed: {e}", exc_info=True)
                return f"I encountered an error running '{cmd.name}': {str(e)}"

        # 2. Fallback to LLM if no command matches
        return await self._fallback_to_llm(text)

    async def _fallback_to_llm(self, text: str) -> str:
        """Use the local LLM when no specific command is triggered."""
        logger.info("No command match. Falling back to LLM.")
        
        # Build chat history for LLM
        messages = [Message(role=m["role"], content=m["content"]) for m in self.session.history]
        messages.append(Message(role="user", content=text))
        
        try:
            response = await self.llm.chat(messages)
            content = response.content
            
            self.session.add_message("user", text)
            self.session.add_message("assistant", content)
            return content
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return f"I'm sorry, my brain is feeling a bit foggy: {str(e)}"
