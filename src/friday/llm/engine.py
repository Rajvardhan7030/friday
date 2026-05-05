"""LLM Engine abstraction layer."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class LLMResponse(BaseModel):
    content: str
    raw_response: Any = None
    usage: Dict[str, int] = Field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None

class LLMEngine(ABC):
    """Abstract Base Class for LLM Engines."""
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """Send a chat completion request."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for a text string."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of strings."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is available and ready."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        pass

    async def aclose(self) -> None:
        """Close any open resources (connections, clients, etc.)."""
        pass
