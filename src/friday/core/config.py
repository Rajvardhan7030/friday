"""Configuration system with YAML and environment variable support."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from platformdirs import user_config_dir

DEFAULT_CONFIG_DIR = Path(user_config_dir("friday"))
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_ENV_PATH = DEFAULT_CONFIG_DIR / ".env"

class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRIDAY_LLM_", extra="ignore")
    
    primary_model: str = "qwen2.5-coder:7b"
    fallback_model: str = "tinyllama"
    ollama_base_url: str = "http://localhost:11434"
    cloud_mode: bool = False
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

class MemoryConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRIDAY_MEMORY_", extra="ignore")
    
    vector_db_path: Path = DEFAULT_CONFIG_DIR / "vector_store"
    sqlite_db_path: Path = DEFAULT_CONFIG_DIR / "friday.db"
    documents_dir: Path = Path.home() / "Documents" / "FridayDocs"
    chunk_size: int = 1000
    chunk_overlap: int = 200

class FridaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRIDAY_", extra="ignore")
    
    config_dir: Path = DEFAULT_CONFIG_DIR
    log_level: str = "INFO"
    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    workspace_dir: Path = DEFAULT_CONFIG_DIR / "workspace"
    persona_name: str = "Friday"
    persona_voice: str = "en_GB-southern_english_female-low"

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "FridaySettings":
        """Load settings from YAML, env, and defaults."""
        config_path = config_path or DEFAULT_CONFIG_PATH
        yaml_config: Dict[str, Any] = {}
        
        if config_path.exists():
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}

        # Merge YAML into settings (environment variables still take precedence)
        return cls(**yaml_config)

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save current settings to YAML atomically."""
        config_path = config_path or DEFAULT_CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize model to dict and save as YAML
        data = self.model_dump(exclude_none=True)
        # Convert Path objects to strings for YAML
        def path_to_str(d):
            for k, v in d.items():
                if isinstance(v, Path):
                    d[k] = str(v)
                elif isinstance(v, dict):
                    path_to_str(v)
        path_to_str(data)
        
        # ATOMIC WRITE: Write to temp file then replace
        fd, temp_path = tempfile.mkstemp(dir=config_path.parent, prefix="config_", suffix=".yaml")
        try:
            with os.fdopen(fd, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            os.replace(temp_path, config_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
