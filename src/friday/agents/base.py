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

class AgentMetadata(BaseModel):
    """Structured metadata for agent results."""
    tts_content: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    trace_id: Optional[str] = None
    execution_time_ms: Optional[int] = None
    retrieval_used: bool = False
    exit_code: Optional[int] = None
    command: Optional[str] = None
    cwd: Optional[str] = None

class AgentResult(BaseModel):
    """Result returned by an agent."""
    content: str
    success: bool = True
    metadata: AgentMetadata = Field(default_factory=AgentMetadata)
    citations: List[Dict[str, str]] = Field(default_factory=list)

class BaseAgent(ABC):
    """Abstract Base Agent all agents must implement."""
    
    def __init__(self, llm_engine: Union[LLMEngine, 'ModelRouter'], config: Optional[Union[Config, Dict[str, Any]]] = None):
        self._llm_engine = llm_engine
        self.config = config or {}

    async def _get_llm(self, task_type: Optional[str] = None) -> LLMEngine:
        """Helper to get the appropriate LLM engine."""
        if hasattr(self._llm_engine, "get_engine_for_task"):
            # It's a ModelRouter
            return await self._llm_engine.get_engine_for_task(task_type or self.name)
        return self._llm_engine

    @property
    def llm(self) -> LLMEngine:
        """Legacy property for backward compatibility (returns raw engine or raises if router)."""
        if hasattr(self._llm_engine, "get_engine_for_task"):
             raise AttributeError("This agent uses a ModelRouter. Use 'await self._get_llm()' instead.")
        return self._llm_engine

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
