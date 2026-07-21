"""Local LLM engine integration using Ollama."""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncIterator
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

    def __init__(self, primary_model: str, fallback_model: str, base_url: str="http://localhost:11434", embedding_model: Optional[str] = None):
        if ollama is None:
            raise LLMError("The 'ollama' package is not installed. Install project dependencies to use the local LLM engine.")
        
        # Guard against using cloud models with the local engine
        cloud_keywords = ["gpt-", "claude-", "gemini-", "mistral-large"]
        if any(kw in primary_model.lower() for kw in cloud_keywords):
            logger.warning(
                f"Model '{primary_model}' looks like a cloud model but you are using the local Ollama engine. "
                "If you intended to use a cloud provider, please run 'friday init' and select 'Cloud API'."
            )
            
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._embedding_model = embedding_model
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
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[LLMResponse, AsyncIterator[LLMResponse]]:
        """Send chat completion to local Ollama."""
        formatted_messages = self._format_messages(messages)
        tried_models: List[str] = []

        # 0. Try known working model first to avoid repeating timeouts/404s for missing primary
        if self._known_chat_model:
            try:
                return await self._chat_with_model(self._known_chat_model, formatted_messages, tools, stream, options)
            except Exception as e:
                if not self._is_model_not_found_error(e):
                    raise
                self._known_chat_model = None

        # 1. Always try primary model first
        try:
            res = await self._chat_with_model(self._primary_model, formatted_messages, tools, stream, options)
            self._known_chat_model = self._primary_model
            return res
        except Exception as e:
            if not self._is_model_not_found_error(e):
                logger.error(f"Ollama chat error with {self._primary_model}: {e}")
                raise LLMError(f"Local LLM engine failed: {e}") from e
            logger.warning(f"Primary model '{self._primary_model}' not found in Ollama.")
            tried_models.append(self._primary_model)

        # 2. Try known working model if it's different and available
        if self._known_chat_model and self._known_chat_model not in tried_models:
            try:
                return await self._chat_with_model(self._known_chat_model, formatted_messages, tools, stream, options)
            except Exception as e:
                if not self._is_model_not_found_error(e):
                    raise
                self._known_chat_model = None

        # 3. Try configured fallback model
        if self._fallback_model and self._fallback_model not in tried_models:
            try:
                res = await self._chat_with_model(self._fallback_model, formatted_messages, tools, stream, options)
                self._known_chat_model = self._fallback_model
                return res
            except Exception as e:
                if not self._is_model_not_found_error(e):
                    logger.error(f"Ollama chat error with {self._fallback_model}: {e}")
                    raise LLMError(f"Local LLM engine failed: {e}") from e
                logger.warning(f"Fallback model '{self._fallback_model}' not found in Ollama.")
                tried_models.append(self._fallback_model)

        # 4. Last resort: any available model
        try:
            available = await self.get_available_models()
            # Filter out models that definitely don't support chat (e.g. embedding models)
            last_resort_models = [
                m for m in available 
                if m not in tried_models and "embed" not in m.lower()
            ]
            
            for model in last_resort_models:
                logger.info(f"Using available model '{model}' as last resort.")
                try:
                    res = await self._chat_with_model(model, formatted_messages, tools, stream, options)
                    self._known_chat_model = model
                    return res
                except Exception as e:
                    err_msg = str(e).lower()
                    # If model doesn't support chat (400), just try the next one instead of failing entirely
                    if "does not support chat" in err_msg or "400" in err_msg:
                        logger.warning(f"Ollama model '{model}' does not support chat. Trying next available.")
                        tried_models.append(model)
                        continue
                        
                    if not self._is_model_not_found_error(e):
                        logger.error(f"Ollama chat error with {model}: {e}")
                        raise LLMError(f"Local LLM engine failed: {e}") from e
                    
                    logger.warning(f"Last-resort model '{model}' not found in Ollama.")
                    tried_models.append(model)
        except LLMError:
            raise
        except Exception as e:
            logger.error(f"Failed to list models for last resort: {e}")

        # Final failure: No models worked
        if not tried_models:
            raise LLMError(
                f"No LLM models found in Ollama. "
                "Please run 'ollama pull llama3' to install a default model."
            )
            
        # If we tried models and they all failed, report the primary and fallback
        raise LLMError(
            f"Configured models ('{self._primary_model}', '{self._fallback_model}') were not found in Ollama, "
            "and no other compatible chat models were available."
        )

    async def get_available_models(self) -> List[str]:
        """List models available in the local Ollama instance."""
        try:
            response = await self._client.list()
            normalized = self._normalize_response(response)
            models = []
            
            raw_models = normalized.get('models', [])
            for m in raw_models:
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

        # 0. Try embedding model if specified
        if self._embedding_model:
            try:
                res = await self._embed_with_model(self._embedding_model, text)
                self._known_embed_model = self._embedding_model
                return res
            except Exception as e:
                if not self._is_model_not_found_error(e):
                    logger.error(f"Ollama embedding error with {self._embedding_model}: {e}")
                    raise LLMError(f"Failed to generate embeddings: {e}") from e
                logger.warning(f"Embedding model '{self._embedding_model}' not found.")
                tried_models.append(self._embedding_model)

        # 1. Always try primary model first
        try:
            res = await self._embed_with_model(self._primary_model, text)
            self._known_embed_model = self._primary_model
            return res
        except Exception as e:
            if not self._is_model_not_found_error(e):
                logger.error(f"Ollama embedding error with {self._primary_model}: {e}")
                raise LLMError(f"Failed to generate embeddings: {e}") from e
            logger.warning(f"Primary model '{self._primary_model}' is not available or does not support embeddings.")
            tried_models.append(self._primary_model)

        # 2. Try known working model if available
        if self._known_embed_model and self._known_embed_model not in tried_models:
            try:
                return await self._embed_with_model(self._known_embed_model, text)
            except Exception as e:
                if not self._is_model_not_found_error(e):
                    raise
                self._known_embed_model = None

        # 3. Try configured fallback model
        if self._fallback_model and self._fallback_model not in tried_models:
            try:
                res = await self._embed_with_model(self._fallback_model, text)
                self._known_embed_model = self._fallback_model
                return res
            except Exception as e:
                if not self._is_model_not_found_error(e):
                    logger.error(f"Ollama embedding error with {self._fallback_model}: {e}")
                    raise LLMError(f"Failed to generate embeddings: {e}") from e
                logger.warning(f"Fallback model '{self._fallback_model}' is not available or does not support embeddings.")
                tried_models.append(self._fallback_model)

        # 4. Last resort
        try:
            available = await self.get_available_models()
            last_resort_models = [m for m in available if m not in tried_models]
            for model in last_resort_models:
                logger.info(f"Trying '{model}' for embeddings as last resort...")
                try:
                    res = await self._embed_with_model(model, text)
                    self._known_embed_model = model
                    return res
                except Exception as e:
                    err_msg = str(e).lower()
                    if "does not support embeddings" in err_msg or "400" in err_msg or "500" in err_msg:
                        logger.warning(f"Ollama model '{model}' does not support embeddings. Trying next available.")
                        tried_models.append(model)
                        continue
                        
                    if not self._is_model_not_found_error(e):
                        logger.error(f"Ollama embedding error with {model}: {e}")
                        raise LLMError(f"Failed to generate embeddings: {e}") from e
                    logger.warning(f"Last-resort embedding model '{model}' not found in Ollama.")
                    tried_models.append(model)
        except Exception as e:
            logger.error(f"Failed to list models for last resort embeddings: {e}")

        # Final failure with clear user guidance
        raise LLMError(
            f"No compatible embedding model available in Ollama (tried: {tried_models}). "
            "Please run 'ollama pull nomic-embed-text' to enable vector memory."
        )

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate local embeddings in batch using concurrency safely without state mutation race conditions."""
        if not texts:
            return []
        
        # Warm up/resolve the embedding model once with the first text to ensure a working model is known
        if not self._known_embed_model:
            await self.embed(texts[0])
            
        # If we have a working model, embed all texts using that resolved model concurrently
        if self._known_embed_model:
            model = self._known_embed_model
            tasks = [self._client.embeddings(model=model, prompt=str(t)) for t in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            output = []
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    # Fallback to individual embed call if a chunk failed
                    output.append(await self.embed(texts[i]))
                else:
                    normalized = self._normalize_response(res)
                    output.append(normalized.get('embedding', []))
            return output

        # Fallback if no model could be pre-resolved
        tasks = [self.embed(text) for text in texts]
        return await asyncio.gather(*tasks)

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to the format expected by Ollama."""
        formatted_messages = []
        for message in messages:
            if hasattr(message, "model_dump"):
                # Use exclude_none=True to avoid sending 'None' fields that Ollama might reject
                formatted_messages.append(message.model_dump(exclude_none=True))
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
        return (
            "not found" in error_msg 
            or "404" in error_msg 
            or "does not support embeddings" in error_msg
            or ("500" in error_msg and "embeddings" in error_msg)
        )

    async def _chat_with_model(
        self,
        model: str,
        formatted_messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[LLMResponse, AsyncIterator[LLMResponse]]:
        # Sanitize options: Ollama uses 'num_predict', not 'max_tokens'
        # Some Ollama versions might reject unknown options if they are strict
        clean_options = None
        if options:
            clean_options = options.copy()
            if "max_tokens" in clean_options:
                if "num_predict" not in clean_options:
                    clean_options["num_predict"] = clean_options["max_tokens"]
                del clean_options["max_tokens"]

        try:
            # Pass model and messages as positional arguments for maximum compatibility
            # and use kwargs for optional ones
            kwargs = {
                "stream": stream,
                "options": clean_options
            }
            if tools:
                kwargs["tools"] = tools

            response = await self._client.chat(
                model,
                messages=formatted_messages,
                **kwargs
            )
        except Exception as e:
            if tools and "does not support tools" in str(e).lower():
                logger.warning(f"Model '{model}' does not support tools. Retrying without tools.")
                response = await self._client.chat(
                    model,
                    messages=formatted_messages,
                    stream=stream,
                    options=clean_options
                )
            else:
                raise

        if stream:
            return self._stream_wrapper(response, model)

        self._current_model = model
        normalized_response = self._normalize_response(response)
        message_data = normalized_response.get('message', {})
        content = message_data.get('content', "")
        tool_calls = message_data.get('tool_calls', None)
        return LLMResponse(content=content, raw_response=normalized_response, usage={}, tool_calls=tool_calls)

    async def _stream_wrapper(self, response_gen: AsyncIterator[Any], model: str) -> AsyncIterator[LLMResponse]:
        """Wrap Ollama stream to yield LLMResponse chunks."""
        self._current_model = model
        async for chunk in response_gen:
            normalized_chunk = self._normalize_response(chunk)
            message_data = normalized_chunk.get('message', {})
            content = message_data.get('content', "")
            tool_calls = message_data.get('tool_calls', None)
            yield LLMResponse(content=content, raw_response=normalized_chunk, tool_calls=tool_calls, is_chunk=True)

    def _normalize_response(self, response: Any) -> Dict[str, Any]:
        """Recursively convert Ollama response (dict or object) to a standard dict format."""
        def to_dict(obj: Any) -> Any:
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            if isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_dict(v) for v in obj]
            if hasattr(obj, '__dict__') and not str(type(obj)).startswith("<class 'httpx."):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            return obj

        result = to_dict(response)
        return result if isinstance(result, dict) else {}

    async def _embed_with_model(self, model: str, text: str) -> List[float]:
        response = await self._client.embeddings(model=model, prompt=text)
        self._current_model = model
        normalized = self._normalize_response(response)
        return normalized.get('embedding', [])

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
