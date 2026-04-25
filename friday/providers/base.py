from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """Send chat messages and return the response."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Return a list of available models for this provider."""
        pass
