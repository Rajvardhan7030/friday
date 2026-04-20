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

    def __init__(self, primary_model: str, fallback_model: str, base_url: str="http://localhost:11434"):
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self.base_url = base_url
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
            formatted_messages = []
            for m in messages:
                if hasattr(m, "model_dump"):
                    formatted_messages.append(m.model_dump())
                elif isinstance(m, dict):
                    formatted_messages.append(m)
                else:
                    raise ValueError(f"Invalid message type: {type(m)}")
            
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
            error_msg = str(e)
            if "not found" in error_msg.lower() or "404" in error_msg:
                logger.warning(f"Model '{self._current_model}' not found in Ollama.")
                
                # Try to find any available model as a last-resort fallback
                if not self._in_fallback:
                    if self._current_model == self._primary_model:
                        logger.info(f"Attempting fallback to {self._fallback_model}")
                        self._current_model = self._fallback_model
                        self._in_fallback = True
                        return await self.chat(messages, tools, stream)
                    else:
                        # We are already in fallback or current model is not primary, 
                        # let's see if ANY model is available
                        available = await self.get_available_models()
                        if available:
                            last_resort = available[0]
                            logger.info(f"Last resort: falling back to available model '{last_resort}'")
                            self._current_model = last_resort
                            self._in_fallback = True # Prevent infinite loop
                            return await self.chat(messages, tools, stream)

            logger.error(f"Ollama chat error with {self._current_model}: {e}")
            # Reset fallback flag for future queries if it failed
            self._in_fallback = False
            
            if "not found" in error_msg.lower() or "404" in error_msg:
                raise LLMError(
                    f"Model '{self._current_model}' not found. "
                    f"Please run 'ollama pull {self._current_model}' to install it."
                )
            raise LLMError(f"Local LLM engine failed: {e}")

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
        try:
            response = await self._client.embeddings(
                model=self._current_model,
                prompt=text
            )
            return response.get('embedding', [])
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "404" in error_msg:
                logger.warning(f"Model '{self._current_model}' not found for embeddings.")
                
                # Try to find any available model
                available = await self.get_available_models()
                if available:
                    last_resort = available[0]
                    logger.info(f"Using '{last_resort}' for embeddings instead.")
                    self._current_model = last_resort
                    return await self.embed(text)

            logger.error(f"Ollama embedding error: {e}")
            if "not found" in error_msg.lower() or "404" in error_msg:
                raise LLMError(
                    f"Model '{self._current_model}' not found. "
                    f"Please run 'ollama pull {self._current_model}' to install it."
                )
            raise LLMError(f"Failed to generate embeddings: {e}")

    async def is_available_async(self) -> bool:
        """Non-blocking health check."""
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(f"{self.base_url}/api/tags", timeout=2.0)
                return r.status_code == 200
            except Exception:
                return False

    def is_available(self) -> bool:
        """Ping Ollama to check availability (not recommended for async paths)."""
        # For legacy compatibility in the CLI
        import httpx as sync_httpx
        try:
            r = sync_httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False
