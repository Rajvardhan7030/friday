"""Abstract Base Agent class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from ..llm.engine import LLMEngine
from ..core.config import Config

class Context(BaseModel):
    """Execution context for agents."""
    user_query: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResult(BaseModel):
    """Result returned by an agent."""
    content: str
    success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    citations: List[Dict[str, str]] = Field(default_factory=list)

class BaseAgent(ABC):
    """Abstract Base Agent all agents must implement."""
    
    def __init__(self, llm_engine: LLMEngine, config: Optional[Union[Config, Dict[str, Any]]] = None):
        self.llm = llm_engine
        self.config = config or {}

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
