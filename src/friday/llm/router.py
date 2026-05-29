"""Multi-Model Routing for Friday."""

import logging
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .engine import LLMEngine
from .api import create_api_engine
from .local import LocalEngine
from ..core.config import Config

logger = logging.getLogger(__name__)

class ModelRoute(BaseModel):
    """Configuration for a specific task route."""
    primary: str
    fallback: Optional[str] = None
    provider: str = "ollama"
    api_base_url: Optional[str] = None
    embedding_model: Optional[str] = None

class ModelPolicy(BaseModel):
    """Collection of routes for different task types."""
    routes: Dict[str, ModelRoute] = Field(default_factory=dict)

class ModelRouter:
    """Selects and initializes LLM engines based on task type and policy."""

    def __init__(self, config: Config):
        self.config = config
        self.policy = self._load_policy()
        self._engines: Dict[str, LLMEngine] = {}

    def _load_policy(self) -> ModelPolicy:
        """Load model routing policy from config."""
        raw_policy = self.config.get("llm.policy", {})
        if not raw_policy:
            # Default policy based on standard config
            primary = self.config.get("llm.primary_model")
            provider = self.config.get("llm.provider", "ollama")
            engine_type = self.config.get("llm.engine", "ollama")
            
            default_route = ModelRoute(
                primary=primary,
                fallback=self.config.get("llm.fallback_model"),
                provider=provider,
                api_base_url=self.config.get("llm.api_base_url") if engine_type == "openai" else None,
                embedding_model=self.config.get("llm.embedding_model")
            )
            
            # Specialized route for summarization to avoid heavy model usage
            summ_primary = self.config.get("llm.summarization_model")
            if summ_primary and summ_primary != primary:
                summ_route = ModelRoute(
                    primary=summ_primary,
                    fallback=primary,
                    provider=provider,
                    api_base_url=default_route.api_base_url
                )
                return ModelPolicy(routes={
                    "default": default_route,
                    "summarization": summ_route
                })
            
            return ModelPolicy(routes={"default": default_route})
        
        return ModelPolicy.model_validate(raw_policy)

    async def get_engine_for_task(self, task_type: str) -> LLMEngine:
        """Get or initialize an LLM engine for a specific task."""
        route = self.policy.routes.get(task_type, self.policy.routes.get("default"))
        if not route:
            raise ValueError(f"No model route defined for task type '{task_type}'")

        engine_key = f"{route.provider}:{route.primary}"
        if engine_key not in self._engines:
            self._engines[engine_key] = self._create_engine(route)
        
        return self._engines[engine_key]

    def _create_engine(self, route: ModelRoute) -> LLMEngine:
        """Initialize an LLM engine based on route configuration."""
        if route.provider == "ollama":
            return LocalEngine(
                primary_model=route.primary,
                fallback_model=route.fallback or route.primary,
                base_url=self.config.get("llm.base_url", "http://localhost:11434"),
                embedding_model=route.embedding_model
            )
        else:
            # API Providers
            return create_api_engine(
                model_name=route.primary,
                api_key=self.config.get(f"llm.api_key"), # Should ideally be more specific
                base_url=route.api_base_url or "https://api.openai.com/v1",
                embedding_model_name=route.embedding_model
            )

    async def aclose(self):
        """Shutdown all initialized engines."""
        for engine in self._engines.values():
            await engine.aclose()
        self._engines.clear()
