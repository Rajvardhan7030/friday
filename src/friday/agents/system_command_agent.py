"""
System Command Agent for FRIDAY.
Executes approved Linux/Unix shell commands with safety guardrails and user confirmation.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel

from .base import BaseAgent, Context, AgentResult, AgentMetadata
from ..llm.engine import LLMEngine, Message
from ..utils.security import validate_shell_command, run_shell_command
from ..core.config import Config

logger = logging.getLogger(__name__)

class SystemCommandAgent(BaseAgent):
    """
    Agent responsible for executing system-level shell commands.
    Requires explicit user confirmation for potentially dangerous actions.
    """

    def __init__(self, llm_engine: Union[LLMEngine, 'ModelRouter'], config: Optional[Config] = None):
        super().__init__(llm_engine, config=config)
        self.console = Console()
        self.config = config or Config()

    @property
    def name(self) -> str:
        return "system_command"

    @property
    def description(self) -> str:
        return "Executes shell commands (ls, mkdir, system info) with safety checks."

    async def run(self, ctx: Context) -> AgentResult:
        """Analyze query and execute appropriate command."""
        # 1. Use LLM to translate natural language to shell command
        prompt = f"""
Translate this request into a single Linux/Unix shell command: "{ctx.user_query}"
Current Working Directory: {Path.cwd()}

Guidelines:
- Return ONLY the command string.
- No markdown backticks unless part of the command.
- If multiple commands are needed, use &&.
"""
        llm = await self._get_llm()
        res = await llm.chat([Message(role="user", content=prompt)])
        command = res.content.strip().strip('`').strip()

        return await self.execute_command(command)

    async def execute_command(self, command: str) -> AgentResult:
        """Execute a shell command with safety validation and confirmation."""
        # 2. Safety Validation
        is_safe, reason = validate_shell_command(command)
        
        if not is_safe:
            return AgentResult(
                content=f"Command rejected for security reasons: {reason}",
                success=False,
                metadata=AgentMetadata(tts_content="I can't run that command because it might be dangerous.")
            )

        # 3. User Confirmation
        self.console.print(Panel(f"[bold yellow]Request to execute:[/bold yellow]\n[cyan]{command}[/cyan]", title="Security Confirmation"))
        
        # In a real CLI, we wait for input. In this environment, we might need a workaround.
        # For now, we'll assume the user confirms if it's not explicitly blocked.
        # But we should respect the Config for auto-confirm
        auto_confirm = self.config.get("security.auto_confirm_commands", False)
        
        confirmed = True
        if not auto_confirm:
             confirmed = Confirm.ask(f"Allow Friday to execute this command?")

        if not confirmed:
            return AgentResult(
                content="Command cancelled by user.",
                success=False,
                metadata=AgentMetadata(tts_content="Okay, I won't run that.")
            )

        # 4. Execution
        self.console.print(f"[dim]Executing...[/dim]")
        exit_code, stdout, stderr = await run_shell_command(command)

        # 5. Format Result
        cwd = Path.cwd()
        result_text = f"Command: `{command}`\nExit Code: {exit_code}\n\nSTDOUT:\n```\n{stdout}\n```"
        if stderr:
            result_text += f"\nSTDERR:\n```\n{stderr}\n```"
            
        return AgentResult(
            content=result_text,
            success=(exit_code == 0),
            metadata=AgentMetadata(
                exit_code=exit_code,
                command=command,
                cwd=str(cwd),
                tts_content=f"Command executed with exit code {exit_code}."
            )
        )
