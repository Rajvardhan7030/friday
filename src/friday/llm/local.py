"""Local LLM engine integration using Ollama."""

import logging
import httpx
import ollama
from typing import List, Dict, Any, Optional
from friday.llm.engine import LLMEngine, Message, LLMResponse
from friday.core.exceptions import LLMError

logger = logging.getLogger(__name__)

class LocalEngine(LLMEngine):
    """Local inference engine using Ollama."""

    def __init__(self, primary_model: str, fallback_model: str, base_url: str):
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._base_url = base_url
        self._client = ollama.AsyncClient(host=base_url)
        self._current_model = primary_model
        self._in_fallback = False

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
            
            content = response.get('message', {}).get('content', "")
            
            return LLMResponse(
                content=content,
                raw_response=response,
                usage={}
            )
        except Exception as e:
            logger.error(f"Ollama chat error with {self._current_model}: {e}")
            # Only fallback if we haven't already done so
            if not self._in_fallback and self._current_model == self._primary_model:
                logger.info(f"Attempting fallback to {self._fallback_model}")
                self._current_model = self._fallback_model
                self._in_fallback = True
                return await self.chat(messages, tools, stream)
            
            # Reset fallback flag for future queries if it failed
            self._in_fallback = False
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

    async def is_available_async(self) -> bool:
        """Non-blocking health check."""
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(f"{self._base_url}/api/tags", timeout=2.0)
                return r.status_code == 200
            except Exception:
                return False

    def is_available(self) -> bool:
        """Ping Ollama to check availability (not recommended for async paths)."""
        # For legacy compatibility in the CLI
        import httpx as sync_httpx
        try:
            r = sync_httpx.get(f"{self._base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False
