"""Custom exceptions for the Friday project."""

from typing import Optional


class FridayError(Exception):
    """Base exception for all Friday errors."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


class ModelNotFoundError(FridayError):
    """Raised when an AI model (TTS, STT, LLM) is missing."""

    def __init__(self, model_name: str, model_path: str, download_hint: Optional[str] = None):
        message = f"Model '{model_name}' not found at {model_path}."
        if download_hint:
            message += f" {download_hint}"
        super().__init__(message)


class SandboxError(FridayError):
    """Raised when code execution in the sandbox fails."""
    pass


class AudioDeviceError(FridayError):
    """Raised when an audio input/output device is unavailable."""
    pass


class ConfigurationError(FridayError):
    """Raised when there is an error in the configuration."""
    pass


class LLMError(FridayError):
    """Raised when an LLM engine fails or a model is missing in the engine."""
    pass


class MemoryInitializationError(FridayError):
    """Raised when memory components (VectorStore, SQLite) fail to initialize."""
    pass


class ToolExecutionError(FridayError):
    """Raised when a tool or MCP command execution fails."""
    pass


class PluginLoadError(FridayError):
    """Raised when a dynamic plugin fails to load."""
    pass


class PermissionDeniedError(FridayError):
    """Raised when a tool execution is blocked by security policy."""
    pass


class BrowserDaemonUnavailable(FridayError):
    """Raised when the Go browser daemon is not reachable."""
    pass


class ProviderRateLimitError(LLMError):
    """Raised when an LLM provider (e.g., Gemini, OpenAI) returns a 429."""
    pass
