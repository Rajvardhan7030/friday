"""Abstract Base Agent class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from src.friday.llm.engine import LLMEngine

class Context(BaseModel):
    """Execution context for agents."""
    user_query: str
    chat_history: List[Dict[str, str]] = []
    metadata: Dict[str, Any] = {}

class AgentResult(BaseModel):
    """Result returned by an agent."""
    content: str
    success: bool = True
    metadata: Dict[str, Any] = {}
    citations: List[Dict[str, str]] = []

class BaseAgent(ABC):
    """Abstract Base Agent all agents must implement."""
    
    def __init__(self, llm_engine: LLMEngine, config: Dict[str, Any] = {}):
        self.llm = llm_engine
        self.config = config

    @abstractmethod
    async def run(self, ctx: Context) -> AgentResult:
        """Execute the agent loop."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the agent's description for the orchestrator."""
        pass
