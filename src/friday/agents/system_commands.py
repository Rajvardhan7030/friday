"""Core system commands registered to the engine."""

from ..core.registry import registry
from ..core.agent_runner import Session

@registry.register(
    name="Help",
    regex=r"^help$|^commands$",
    description="Show all available commands",
    usage="help"
)
async def help_handler(session: Session, **_kwargs):
    """Returns the help text for all registered commands."""
    return registry.get_help()

@registry.register(
    name="Greeting",
    regex=r"\b(?:hi+|hello+|hey+|greetings|morning|evening)\b",
    description="Respond to greetings",
    usage="hello"
)
async def greeting_handler(session: Session, **_kwargs):
    """Responds to common greetings."""
    return "Hello! I am FRIDAY. How can I help you today?"

@registry.register(
    name="Identity",
    regex=r"who (?:are|r) (?:you|u)\??|your name",
    description="Ask about Friday's identity",
    usage="who are you?",
    priority=10
)
async def identity_handler(session: Session, **_kwargs):
    """Returns a short description of Friday."""
    return "I am FRIDAY (Female Replacement Intelligent Digital Assistant Youth), your personal AI assistant. I run locally and prioritize your privacy."

@registry.register(
    name="Clear",
    regex=r"^(?:clear|reset|delete)(?: (?:history|session|chat))?$",
    description="Clear the current conversation history",
    usage="clear or clear history"
)
async def clear_handler(session: Session, **_kwargs):
    """Resets the session history."""
    session.history = []
    return "Conversation history cleared."
