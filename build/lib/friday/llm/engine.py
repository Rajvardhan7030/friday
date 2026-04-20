"""LLM Engine abstraction layer."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class LLMResponse(BaseModel):
    content: str
    raw_response: Any = None
    usage: Dict[str, int] = {}

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
    def is_available(self) -> bool:
        """Check if the engine is available and ready."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        pass
