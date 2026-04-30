"""LLM engine integration for OpenAI-compatible APIs."""

import logging
from typing import List, Dict, Any, Optional
from .engine import LLMEngine, Message, LLMResponse
from ..core.exceptions import LLMError

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)

class APIEngine(LLMEngine):
    """Remote inference engine using OpenAI-compatible API."""

    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        if httpx is None:
            raise LLMError("The 'httpx' package is not installed. Install project dependencies to use the API engine.")
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._api_key}"},
            base_url=self._base_url,
            timeout=60.0
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    async def chat(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """Send chat completion to OpenAI-compatible API."""
        if stream:
            # Streaming not yet implemented in this basic version
            logger.warning("Streaming requested but not implemented for APIEngine.")

        url = "/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": [m.model_dump() for m in messages],
        }
        if tools:
            payload["tools"] = tools

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content", "")
            tool_calls = message.get("tool_calls")
            
            usage = data.get("usage", {})
            
            return LLMResponse(
                content=content or "",
                raw_response=data,
                usage=usage,
                tool_calls=tool_calls
            )
        except Exception as e:
            logger.error(f"API chat error: {e}")
            raise LLMError(f"API LLM engine failed: {e}")

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI-compatible API."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch using OpenAI-compatible API."""
        url = "/embeddings"
        payload = {
            "model": "text-embedding-3-small" if "gpt" in self._model_name else self._model_name,
            "input": texts,
        }

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Sort by index to ensure order matches input
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as e:
            logger.error(f"API embedding error: {e}")
            raise LLMError(f"Failed to generate API embeddings: {e}")

    def is_available(self) -> bool:
        """Check if the API is reachable (basic check)."""
        # We don't want to make an authenticated call here just for a check
        # maybe just check if we have a key
        return bool(self._api_key)

    async def is_available_async(self) -> bool:
        """Non-blocking health check."""
        return self.is_available()
