"""Core system commands registered to the engine."""

from ..core.registry import registry
from ..core.agent_runner import Session

@registry.register(
    name="Help",
    regex=r"^help$|^commands$",
    description="Show all available commands",
    usage="help"
)
async def help_handler(session: Session, **kwargs):
    """Returns the help text for all registered commands."""
    return registry.get_help()

@registry.register(
    name="Greeting",
    regex=r"^(?:hi+|hello+|hey+|greetings|good (?:morning|evening)|morning|evening)(?:\s+friday)?[!.?]*$",
    description="Respond to greetings",
    usage="hello"
)
async def greeting_handler(session: Session, **kwargs):
    """Responds to common greetings."""
    return "Hello! I am FRIDAY. How can I help you today?"

@registry.register(
    name="Identity",
    regex=r"^(?:who (?:are|r) (?:you|u)|what(?:'s| is) your name|your name)[?.!]*$",
    description="Ask about Friday's identity",
    usage="who are you?",
    priority=10
)
async def identity_handler(session: Session, **kwargs):
    """Returns a short description of Friday."""
    return "I am FRIDAY (Female Replacement Intelligent Digital Assistant Youth), your personal AI assistant. I run locally and prioritize your privacy."

@registry.register(
    name="Clear",
    regex=r"^(?:clear|reset|delete)(?: (?:history|session|chat))?$",
    description="Clear the current conversation history",
    usage="clear or clear history"
)
async def clear_handler(session: Session, **kwargs):
    """Resets the session history."""
    session.history = []
    return "Conversation history cleared."

@registry.register(
    name="RunCommand",
    regex=r"^(?:/run|run)\s+(.+)$",
    description="Execute a shell command with confirmation",
    usage="/run <command> or run <command>"
)
async def run_command_handler(session: Session, command: str, llm=None, config=None, **kwargs):
    """Delegates to SystemCommandAgent to execute a direct shell command."""
    from .system_command_agent import SystemCommandAgent
    agent = SystemCommandAgent(llm, config)
    result = await agent.execute_command(command)
    return result.content

@registry.register(
    name="ScriptCommand",
    regex=r"^(?:/script|script)\s+(.+)$",
    description="Generate and execute a Python script in the sandbox",
    usage="/script <description> or script <description>"
)
async def script_command_handler(session: Session, description: str, llm=None, config=None, **kwargs):
    """Delegates to CodeAssistantAgent to generate and run a script."""
    from .code_assistant import CodeAssistantAgent
    from .sandbox_executor import SandboxExecutor
    from .base import Context
    
    agent = CodeAssistantAgent(llm, SandboxExecutor(config))
    ctx = Context(user_query=description)
    
    # Optional: We could stream "Planning..." "Generating..." if we had a way to callback
    # For now, just run it.
    result = await agent.run(ctx)
    return result.content
