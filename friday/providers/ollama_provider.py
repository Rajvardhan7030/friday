import ollama
from typing import List, Optional, Dict, Any
from .base import BaseLLMProvider
import logging

logger = logging.getLogger(__name__)

class OllamaProvider(BaseLLMProvider):
    def __init__(self, default_model: str = "llama3", base_url: Optional[str] = None):
        if base_url:
            self.client = ollama.AsyncClient(host=base_url)
        else:
            self.client = ollama.AsyncClient()
        self.default_model = default_model

    async def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        model_name = model or self.default_model
        try:
            response = await self.client.chat(
                model=model_name,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    async def embed(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings(
                model=self.default_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Ollama embed error: {e}")
            raise

    @property
    def available_models(self) -> List[str]:
        return ["llama3", "mistral", "phi3"]
