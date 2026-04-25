import openai
from typing import List, Optional, Dict, Any
from .base import BaseLLMProvider
import logging

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, default_model: str = "gpt-4-turbo", base_url: Optional[str] = None):
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
            logger.error(f"OpenAI chat error: {e}")
            raise

    async def embed(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embed error: {e}")
            raise

    @property
    def available_models(self) -> List[str]:
        return ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
