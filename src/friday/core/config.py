"""Central configuration management for Friday."""

import os
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class Config:
    """Handles loading and accessing configuration from yaml."""

    DEFAULT_BASE_DIR = Path.home() / ".friday"
    DEFAULT_CONFIG_PATH = DEFAULT_BASE_DIR / "config.yaml"

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration from a file or use defaults."""
        self.base_dir = self.DEFAULT_BASE_DIR
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._data: Dict[str, Any] = self._get_default_values()

        if self.config_path.exists():
            self._load_config()
        else:
            self._save_default_config()

        self._ensure_directories()

    def _get_default_values(self) -> Dict[str, Any]:
        """Return a dictionary of default configuration values."""
        return {
            "logging": {
                "level": "INFO",
                "file": str(self.base_dir / "logs" / "friday.log"),
                "rotate_daily": True,
            },
            "voice": {
                "tts": {
                    "model": "en_GB-vits-low",
                    "model_path": str(self.base_dir / "models" / "en_GB-vits-low.onnx"),
                    "piper_path": "piper",
                },
                "stt": {
                    "model_path": str(self.base_dir / "models" / "vosk-model-small-en-us-0.15"),
                    "samplerate": 16000,
                    "timeout": 10,
                    "silence_limit": 2,
                    "energy_threshold": 300,
                    "device_index": None,
                },
            },
            "security": {
                "sandbox_timeout": 30,
                "sandbox_backend": "unshare",  # or 'docker', 'none'
            },
        }

    def _load_config(self) -> None:
        """Load configuration from the yaml file."""
        try:
            with open(self.config_path, "r") as f:
                loaded_data = yaml.safe_load(f)
                if loaded_data:
                    self._deep_update(self._data, loaded_data)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise ConfigurationError(f"Error reading configuration file: {e}")

    def _save_default_config(self) -> None:
        """Save the default configuration to a file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(self._data, f, default_flow_style=False)
            logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not save default config to {self.config_path}: {e}")

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.base_dir / "models",
            self.base_dir / "logs",
            self.base_dir / "workspace",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update a dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a value using a dot-separated path (e.g., 'voice.tts.model')."""
        keys = key_path.split(".")
        value = self._data
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


def download_model(url: str, dest: Union[str, Path]) -> None:
    """Stub for model auto-download with progress bar."""
    import requests
    from tqdm import tqdm

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model from {url} to {dest_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
