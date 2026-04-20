"""Custom exception hierarchy for the FRIDAY framework."""

class FridayError(Exception):
    """Base exception for all FRIDAY errors."""
    pass

class HardwareError(FridayError):
    """Raised when hardware detection or requirement check fails."""
    pass

class ConfigError(FridayError):
    """Raised when configuration is invalid or missing."""
    pass

class LLMError(FridayError):
    """Raised when LLM engine operations fail."""
    pass

class MemoryError(FridayError):
    """Raised when vector store or conversation history operations fail."""
    pass

class SkillError(FridayError):
    """Raised when a skill fails to load or execute."""
    pass

class AgentError(FridayError):
    """Raised when agent execution fails."""
    pass

class SecurityError(FridayError):
    """Raised when a security boundary is violated."""
    pass
