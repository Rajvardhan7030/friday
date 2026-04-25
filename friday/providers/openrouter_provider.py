import openai
from typing import List, Optional, Dict, Any
from .base import BaseLLMProvider
import logging

logger = logging.getLogger(__name__)

class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, api_key: str, default_model: str = "google/gemini-2.5-flash-preview", base_url: Optional[str] = "https://openrouter.ai/api/v1"):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model

    async def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        model_name = model or self.default_model
        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenRouter chat error: {e}")
            raise

    async def embed(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                model="openai/text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenRouter embed error, falling back to mock zeros: {e}")
            return [0.0] * 1536

    @property
    def available_models(self) -> List[str]:
        return ["google/gemini-2.5-flash-preview", "anthropic/claude-3-opus", "meta-llama/llama-3-70b-instruct"]
