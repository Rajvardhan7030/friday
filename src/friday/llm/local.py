"""Local LLM engine integration using Ollama."""

import logging
from typing import List, Dict, Any, Optional
from .engine import LLMEngine, Message, LLMResponse
from ..core.exceptions import LLMError

try:
    import httpx
except ImportError:
    httpx = None

try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger(__name__)

class LocalEngine(LLMEngine):
    """Local inference engine using Ollama."""

    def __init__(self, primary_model: str, fallback_model: str, base_url: str="http://localhost:11434"):
        if ollama is None:
            raise LLMError("The 'ollama' package is not installed. Install project dependencies to use the local LLM engine.")
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self.base_url = base_url
        self._client = ollama.AsyncClient(host=base_url)
        self._current_model = primary_model
        self._known_chat_model: Optional[str] = None
        self._known_embed_model: Optional[str] = None

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
        formatted_messages = self._format_messages(messages)
        tried_models: List[str] = []

        # 1. Try known working model first if available
        if self._known_chat_model:
            try:
                return await self._chat_with_model(self._known_chat_model, formatted_messages, tools, stream)
            except Exception as e:
                if not self._is_model_not_found_error(e):
                    raise
                self._known_chat_model = None

        # 2. Try configured models
        try:
            for model in self._model_attempt_order():
                try:
                    res = await self._chat_with_model(model, formatted_messages, tools, stream)
                    self._known_chat_model = model
                    return res
                except Exception as e:
                    if not self._is_model_not_found_error(e):
                        logger.error(f"Ollama chat error with {model}: {e}")
                        raise LLMError(f"Local LLM engine failed: {e}") from e

                    logger.warning(f"Model '{model}' not found in Ollama.")
                    tried_models.append(model)

            # 3. Last resort: any available model
            available = await self.get_available_models()
            last_resort_models = [m for m in available if m not in tried_models]
            for model in last_resort_models:
                logger.info(f"Using available model '{model}' as last resort.")
                try:
                    res = await self._chat_with_model(model, formatted_messages, tools, stream)
                    self._known_chat_model = model
                    return res
                except Exception as e:
                    if not self._is_model_not_found_error(e):
                        logger.error(f"Ollama chat error with {model}: {e}")
                        raise LLMError(f"Local LLM engine failed: {e}") from e
                    logger.warning(f"Last-resort model '{model}' not found in Ollama.")
                    tried_models.append(model)

            missing = tried_models or [self._primary_model]
            raise LLMError(
                f"Model '{missing[-1]}' not found. "
                f"Please run 'ollama pull {missing[-1]}' to install it."
            )
        except LLMError:
            raise

    async def get_available_models(self) -> List[str]:
        """List models available in the local Ollama instance."""
        try:
            response = await self._client.list()
            models = []
            
            # Handle both object-based and dict-based responses from different library versions
            raw_models = getattr(response, 'models', []) if hasattr(response, 'models') else response.get('models', [])
            
            for m in raw_models:
                name = None
                if hasattr(m, 'model'): # Newer versions
                    name = m.model
                elif hasattr(m, 'name'): # Possible variation
                    name = m.name
                elif isinstance(m, dict):
                    name = m.get('model') or m.get('name')
                
                if name:
                    models.append(name)
            return models
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def embed(self, text: str) -> List[float]:
        """Generate local embeddings."""
        tried_models: List[str] = []

        # 1. Try known working model first
        if self._known_embed_model:
            try:
                return await self._embed_with_model(self._known_embed_model, text)
            except Exception as e:
                if not self._is_model_not_found_error(e):
                    raise
                self._known_embed_model = None

        # 2. Try configured models
        try:
            for model in self._model_attempt_order():
                try:
                    res = await self._embed_with_model(model, text)
                    self._known_embed_model = model
                    return res
                except Exception as e:
                    if not self._is_model_not_found_error(e):
                        logger.error(f"Ollama embedding error with {model}: {e}")
                        raise LLMError(f"Failed to generate embeddings: {e}") from e

                    logger.warning(f"Model '{model}' not found for embeddings.")
                    tried_models.append(model)

            # 3. Last resort
            available = await self.get_available_models()
            last_resort_models = [m for m in available if m not in tried_models]
            for model in last_resort_models:
                logger.info(f"Using '{model}' for embeddings instead.")
                try:
                    res = await self._embed_with_model(model, text)
                    self._known_embed_model = model
                    return res
                except Exception as e:
                    if not self._is_model_not_found_error(e):
                        logger.error(f"Ollama embedding error with {model}: {e}")
                        raise LLMError(f"Failed to generate embeddings: {e}") from e
                    logger.warning(f"Last-resort embedding model '{model}' not found in Ollama.")
                    tried_models.append(model)

            missing = tried_models or [self._primary_model]
            raise LLMError(
                f"Model '{missing[-1]}' not found. "
                f"Please run 'ollama pull {missing[-1]}' to install it."
            )
        except LLMError:
            raise

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to the format expected by Ollama."""
        formatted_messages = []
        for message in messages:
            if hasattr(message, "model_dump"):
                formatted_messages.append(message.model_dump())
            elif isinstance(message, dict):
                formatted_messages.append(message)
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        return formatted_messages

    def _model_attempt_order(self) -> List[str]:
        """Return the preferred model order for a fresh request."""
        models = [self._primary_model]
        if self._fallback_model and self._fallback_model != self._primary_model:
            models.append(self._fallback_model)
        return models

    @staticmethod
    def _is_model_not_found_error(error: Exception) -> bool:
        error_msg = str(error).lower()
        return "not found" in error_msg or "404" in error_msg

    async def _chat_with_model(
        self,
        model: str,
        formatted_messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
    ) -> LLMResponse:
        response = await self._client.chat(
            model=model,
            messages=formatted_messages,
            tools=tools,
            stream=stream
        )
        self._current_model = model
        content = response.get('message', {}).get('content', "")
        return LLMResponse(content=content, raw_response=response, usage={})

    async def _embed_with_model(self, model: str, text: str) -> List[float]:
        response = await self._client.embeddings(model=model, prompt=text)
        self._current_model = model
        return response.get('embedding', [])

    async def is_available_async(self) -> bool:
        """Non-blocking health check."""
        if httpx is None:
            return False
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(f"{self.base_url}/api/tags", timeout=2.0)
                return r.status_code == 200
            except Exception:
                return False

    def is_available(self) -> bool:
        """Ping Ollama to check availability (not recommended for async paths)."""
        if httpx is None:
            return False
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False
