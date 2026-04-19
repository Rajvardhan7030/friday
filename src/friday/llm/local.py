"""Local LLM engine integration using Ollama."""

import logging
from typing import List, Dict, Any, Optional
import ollama
from src.friday.llm.engine import LLMEngine, Message, LLMResponse
from src.friday.core.exceptions import LLMError

logger = logging.getLogger(__name__)

class LocalEngine(LLMEngine):
    """Local inference engine using Ollama."""

    def __init__(self, primary_model: str, fallback_model: str, base_url: str):
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._client = ollama.AsyncClient(host=base_url)
        self._current_model = primary_model

    @property
    def model_name(self) -> str:
        return self._current_model

    async def chat(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """Send chat completion to local Ollama."""
        try:
            # Convert Message objects to dicts for Ollama
            formatted_messages = [m.model_dump() for m in messages]
            
            response = await self._client.chat(
                model=self._current_model,
                messages=formatted_messages,
                tools=tools,
                stream=stream
            )
            
            # Check if tools were called but we don't have full ToolCall handling yet
            # In local v0.1 we'll handle content first
            content = response.get('message', {}).get('content', "")
            
            return LLMResponse(
                content=content,
                raw_response=response,
                usage={} # Ollama v0.1 client doesn't always provide usage
            )
        except Exception as e:
            logger.error(f"Ollama chat error with {self._current_model}: {e}")
            if self._current_model == self._primary_model:
                logger.info(f"Attempting fallback to {self._fallback_model}")
                self._current_model = self._fallback_model
                return await self.chat(messages, tools, stream)
            raise LLMError(f"Local LLM engine failed: {e}")

    async def embed(self, text: str) -> List[float]:
        """Generate local embeddings."""
        try:
            response = await self._client.embeddings(
                model=self._current_model,
                prompt=text
            )
            return response.get('embedding', [])
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise LLMError(f"Failed to generate embeddings: {e}")

    def is_available(self) -> bool:
        """Ping Ollama to check availability."""
        # This is a bit of a sync check in an async engine
        # In a real app we'd want a more robust health check
        import httpx
        try:
            r = httpx.get(f"{self._client._host}/api/tags")
            return r.status_code == 200
        except Exception:
            return False
