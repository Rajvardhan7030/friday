"""Central configuration management for Friday."""

import os
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class Config:
    """Handles loading and accessing configuration from yaml with environment overrides."""

    DEFAULT_BASE_DIR = Path.home() / ".friday"
    DEFAULT_CONFIG_PATH = DEFAULT_BASE_DIR / "config.yaml"

    # Provider defaults: (Engine, Model Name, Base URL, Embedding Model)
    PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
        "ollama": {
            "engine": "ollama",
            "model": "llama3",
            "url": "http://localhost:11434",
            "embedding": "nomic-embed-text:latest"
        },
        "openai": {
            "engine": "openai",
            "model": "gpt-4o",
            "url": "https://api.openai.com/v1",
            "embedding": "text-embedding-3-small"
        },
        "gemini": {
            "engine": "openai",
            "model": "gemini-2.5-flash",
            "url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "embedding": "gemini-embedding-001"
        },
        "mistral": {
            "engine": "openai",
            "model": "mistral-large-latest",
            "url": "https://api.mistral.ai/v1",
            "embedding": "mistral-embed"
        },
        "groq": {
            "engine": "openai",
            "model": "llama3-70b-8192",
            "url": "https://api.groq.com/openai/v1",
            "embedding": "text-embedding-3-small"
        },
        "openrouter": {
            "engine": "openai",
            "model": "anthropic/claude-3.5-sonnet",
            "url": "https://openrouter.ai/api/v1",
            "embedding": "text-embedding-3-small"
        },
        "other": {
            "engine": "openai",
            "model": "gpt-4o",
            "url": "https://api.openai.com/v1",
            "embedding": "text-embedding-3-small"
        },
    }

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration from a file or use defaults."""
        self.base_dir = self.DEFAULT_BASE_DIR
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._data: Dict[str, Any] = self._get_default_values()

        if self.config_path.exists():
            self._load_config()
        else:
            self._save_default_config()

        # Overwrite with environment variables (e.g., FRIDAY_LLM_PRIMARY_MODEL)
        self._load_env_overrides()
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
                    "model": "en_GB-jenny_dioco-medium",
                    "model_path": str(self.base_dir / "models" / "en_GB-jenny_dioco-medium.onnx"),
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
                "urls": {
                    "tts": {
                        "en_GB-jenny_dioco-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
                        "en_GB-alan-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx"
                    },
                    "stt": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
                },
                "hashes": {
                    "tts": {
                        "en_GB-jenny_dioco-medium": "", # SHA256 hashes go here
                        "en_GB-alan-medium": ""
                    },
                    "stt": ""
                }
            },
            "security": {
                "sandbox_timeout": 30,
                "sandbox_backend": "unshare",  # or 'docker', 'none'
                "shell_command_timeout": 30,
                "shell_command_allow_sudo": False,
                "shell_command_blocked_patterns": [
                    "rm -rf /",
                    "mkfs",
                    "dd if=/dev/zero",
                    ":(){ :|:& };:"
                ],
            },
            "session": {
                "max_history_messages": 100,
                "recent_messages": 20,
                "summary_max_chars": 4000,
            },
            "agents": {
                "skip_modules": [],
            },
            "memory": {
                "enabled": True,
                "persist_directory": str(self.base_dir / "memory_store"),
                "auto_index_directories": [str(self.base_dir / "workspace")],
                "retrieval_limit": 3,
                "auto_remember_conversations": True,
            },
            "llm": {
                "engine": "ollama",  # "ollama" or "openai"
                "primary_model": "mistral:latest",
                "fallback_model": "llama3:latest",
                "embedding_model": "nomic-embed-text:latest",
                "base_url": "http://localhost:11434",
                "api_key": "",
                "api_base_url": "https://api.openai.com/v1",
            },
            "mcp_servers": {
            },
            "skills": {
                "browser": {
                    "daemon_url": "http://localhost:9000"
                }
            }
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
            # Use os.open with 0o600 to ensure restricted permissions from creation
            fd = os.open(self.config_path, os.O_CREAT | os.O_WRONLY, 0o600)
            with open(fd, "w") as f:
                yaml.dump(self._data, f, default_flow_style=False)
            logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not save default config to {self.config_path}: {e}")

    def _load_env_overrides(self) -> None:
        """Override config with environment variables prefixed with FRIDAY_."""
        for key, value in os.environ.items():
            if key.startswith("FRIDAY_"):
                # Convert FRIDAY_VOICE_TTS_MODEL to voice.tts.model
                config_key = key[7:].lower().replace("_", ".")
                try:
                    self.set(config_key, value, save=False)
                except ConfigurationError:
                    # Ignore env variables that don't match our schema
                    pass

    def _ensure_directories(self) -> None:
        """Create necessary directories and files with correct permissions."""
        directories = [
            self.base_dir / "models",
            self.base_dir / "logs",
            self.base_dir / "workspace",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Enforce log file permissions early
        log_file = Path(self.get("logging.file"))
        try:
            if not log_file.exists():
                log_file.parent.mkdir(parents=True, exist_ok=True)
                # Create empty file with 0o600
                fd = os.open(log_file, os.O_CREAT | os.O_WRONLY, 0o600)
                os.close(fd)
            else:
                os.chmod(log_file, 0o600)
        except Exception as e:
            logger.warning(f"Could not secure log file permissions: {e}")

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

    def set(self, key_path: str, value: Any, save: bool = True) -> None:
        """Set a value using a dot-separated path and optionally save to disk."""
        keys = key_path.split(".")
        data = self._data
        
        # Type casting based on default values
        current_val = self.get(key_path)
        if current_val is not None:
            try:
                if isinstance(current_val, bool):
                    if isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes")
                elif isinstance(current_val, int):
                    value = int(value)
                elif isinstance(current_val, float):
                    value = float(value)
            except (ValueError, TypeError):
                raise ConfigurationError(f"Invalid value for {key_path}. Expected {type(current_val).__name__}.")

        # Navigate/Create nested dicts
        for key in keys[:-1]:
            if key not in data or not isinstance(data[key], dict):
                data[key] = {}
            data = data[key]
        
        data[keys[-1]] = value
        
        if save:
            self.save()

    def save(self) -> None:
        """Persist the current configuration to the YAML file with restricted permissions."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            # Create file with 0o600 if it doesn't exist, or update permissions if it does
            mode = 0o600
            if not self.config_path.exists():
                with open(os.open(self.config_path, os.O_CREAT | os.O_WRONLY, mode), "w") as f:
                    yaml.dump(self._data, f, default_flow_style=False)
            else:
                with open(self.config_path, "w") as f:
                    yaml.dump(self._data, f, default_flow_style=False)
                os.chmod(self.config_path, mode)
                
            logger.info(f"Configuration saved to {self.config_path} (mode 0o600)")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def get_all(self) -> Dict[str, Any]:
        """Return the entire configuration dictionary."""
        return self._data


def download_model(url: str, dest: Union[str, Path], expected_hash: Optional[str] = None) -> None:
    """Download a file with progress bar and optional hash verification."""
    import requests
    import hashlib
    from tqdm import tqdm

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary file for downloading
    temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")

    logger.info(f"Downloading model from {url} to {dest_path}")
    
    sha256 = hashlib.sha256() if expected_hash else None
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(temp_dest, "wb") as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    if sha256:
                        sha256.update(chunk)
                    bar.update(len(chunk))
        
        if expected_hash:
            actual_hash = sha256.hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Hash verification failed for {url}. Expected {expected_hash}, got {actual_hash}")
        
        # Atomic rename
        temp_dest.replace(dest_path)
        logger.info(f"Successfully downloaded and verified {dest_path.name}")

    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        raise e
