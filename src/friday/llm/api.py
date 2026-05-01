"""LLM engine integration for OpenAI-compatible APIs."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from .engine import LLMEngine, Message, LLMResponse
from ..core.exceptions import LLMError

try:
    import httpx
except ImportError:
    httpx = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log
except ImportError:
    # Fallback if tenacity is not available
    logging.getLogger(__name__).warning("The 'tenacity' package is not installed. Retries will be disabled for API calls.")
    def retry(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    stop_after_attempt = wait_exponential = retry_if_exception = before_sleep_log = lambda *args, **kwargs: None

logger = logging.getLogger(__name__)

def _is_retryable_api_error(e: Exception) -> bool:
    """Check if the error is a transient API error that should be retried."""
    if isinstance(e, httpx.HTTPStatusError):
        # Retry on Rate Limit (429) or Server Errors (5xx)
        return e.response.status_code == 429 or e.response.status_code >= 500
    if isinstance(e, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, httpx.WriteTimeout)):
        # Retry on connection issues or timeouts
        return True
    return False

class APIEngine(LLMEngine):
    """Remote inference engine using OpenAI-compatible API."""

    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        base_url: str = "https://api.openai.com/v1",
        embedding_model_name: Optional[str] = None
    ):
        if httpx is None:
            raise LLMError("The 'httpx' package is not installed. Install project dependencies to use the API engine.")
        
        self._model_name = model_name
        self._embedding_model_name = embedding_model_name
        self._api_key = api_key
        if not self._api_key:
            logger.warning("No API key provided for APIEngine. Remote calls will fail.")
        self._base_url = base_url.rstrip("/") + "/"
        
        headers, params = self._get_auth_config()

        self._client = httpx.AsyncClient(
            headers=headers,
            params=params,
            base_url=self._base_url,
            timeout=60.0
        )

    def _get_auth_config(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return headers and params for authentication."""
        return {"Authorization": f"Bearer {self._api_key}"}, {}

    @property
    def model_name(self) -> str:
        return self._model_name

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception(_is_retryable_api_error),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _request(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Internal helper to make requests with retries."""
        # Ensure url is relative (no leading slash) to join correctly with base_url
        url = url.lstrip("/")
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def chat(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """Send chat completion to OpenAI-compatible API."""
        if stream:
            raise NotImplementedError("Streaming is not yet implemented for APIEngine.")

        url = "chat/completions"
        payload = {
            "model": self._model_name,
            "messages": [m.model_dump() for m in messages],
        }
        if tools:
            payload["tools"] = tools

        try:
            data = await self._request(url, payload)
            
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
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise LLMError(f"Authentication failed (401/403). Please check your API key in config.yaml.") from e
            raise LLMError(f"API LLM engine failed: {e}")
        except Exception as e:
            raise LLMError(f"API LLM engine failed: {e}")

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI-compatible API."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch using OpenAI-compatible API."""
        url = "embeddings"
        
        # Determine which embedding model to use
        model = self._get_embedding_model()

        payload = {
            "model": model,
            "input": texts,
        }

        try:
            data = await self._request(url, payload)
            # Sort by index to ensure order matches input
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise LLMError(f"Authentication failed (401/403). Please check your API key in config.yaml.") from e
            raise LLMError(f"Failed to generate API embeddings: {e}")
        except Exception as e:
            raise LLMError(f"Failed to generate API embeddings: {e}")

    def _get_embedding_model(self) -> str:
        """Heuristic to choose a default embedding model if none specified."""
        if self._embedding_model_name:
            return self._embedding_model_name
        return "text-embedding-3-small" if "gpt" in self._model_name else self._model_name

    def is_available(self) -> bool:
        """Check if the API is reachable (basic check)."""
        return bool(self._api_key)

    async def is_available_async(self) -> bool:
        """Non-blocking health check."""
        return self.is_available()

class GeminiEngine(APIEngine):
    """Specialized engine for Google Gemini API via OpenAI-compatible endpoint."""

    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        embedding_model_name: Optional[str] = None
    ):
        # Auto-correct Gemini base URL if it's missing the OpenAI compatibility path
        if "generativelanguage.googleapis.com" in base_url and "/openai" not in base_url:
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            # Only append if the path is empty or just a slash (i.e., user only provided the domain)
            if parsed.path in ("", "/"):
                base_url = base_url.rstrip("/") + "/v1beta/openai"
        
        # Normalize model names
        if ":" in model_name or model_name in ("gemini", "gemini3", "default"):
            model_name = "gemini-1.5-flash"
            
        if embedding_model_name and (":" in embedding_model_name or "nomic" in embedding_model_name):
            embedding_model_name = "text-embedding-004"

        super().__init__(model_name, api_key, base_url, embedding_model_name)

    def _get_auth_config(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        # Google AI Studio prefers API key in query param or x-goog-api-key header
        # We use the header only to avoid exposure in logs/proxies via query params
        return {"x-goog-api-key": self._api_key}, {}

    def _get_embedding_model(self) -> str:
        if self._embedding_model_name:
            return self._embedding_model_name
        return "text-embedding-004"

class MistralEngine(APIEngine):
    """Specialized engine for Mistral AI API."""

    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        base_url: str = "https://api.mistral.ai/v1",
        embedding_model_name: Optional[str] = None
    ):
        super().__init__(model_name, api_key, base_url, embedding_model_name)

    def _get_embedding_model(self) -> str:
        if self._embedding_model_name:
            return self._embedding_model_name
        return "mistral-embed"

class GroqEngine(APIEngine):
    """Specialized engine for Groq API."""

    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        base_url: str = "https://api.groq.com/openai/v1",
        embedding_model_name: Optional[str] = None
    ):
        super().__init__(model_name, api_key, base_url, embedding_model_name)

    def _get_embedding_model(self) -> str:
        # Groq doesn't host embeddings yet, fallback to a sensible default or raise
        if self._embedding_model_name:
            return self._embedding_model_name
        return "text-embedding-3-small" # Heuristic fallback

class OpenRouterEngine(APIEngine):
    """Specialized engine for OpenRouter API."""

    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        base_url: str = "https://openrouter.ai/api/v1",
        embedding_model_name: Optional[str] = None
    ):
        super().__init__(model_name, api_key, base_url, embedding_model_name)
    
    def _get_auth_config(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        headers, params = super()._get_auth_config()
        # OpenRouter likes these for their rankings
        headers["HTTP-Referer"] = "https://github.com/google-gemini/friday"
        headers["X-Title"] = "FRIDAY AI"
        return headers, params

def create_api_engine(
    model_name: str, 
    api_key: str, 
    base_url: str = "https://api.openai.com/v1",
    embedding_model_name: Optional[str] = None
) -> APIEngine:
    """Factory to create the appropriate API engine based on the base URL."""
    url = base_url.lower()
    if "generativelanguage.googleapis.com" in url:
        return GeminiEngine(model_name, api_key, base_url, embedding_model_name)
    if "mistral.ai" in url:
        return MistralEngine(model_name, api_key, base_url, embedding_model_name)
    if "groq.com" in url:
        return GroqEngine(model_name, api_key, base_url, embedding_model_name)
    if "openrouter.ai" in url:
        return OpenRouterEngine(model_name, api_key, base_url, embedding_model_name)
    
    return APIEngine(model_name, api_key, base_url, embedding_model_name)
