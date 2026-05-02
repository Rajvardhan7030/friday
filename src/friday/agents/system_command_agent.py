"""
System Command Agent for FRIDAY.
Executes approved Linux/Unix shell commands with safety guardrails and user confirmation.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel

from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message
from ..utils.security import validate_shell_command, run_shell_command
from ..core.config import Config

logger = logging.getLogger(__name__)
console = Console()

class SystemCommandAgent(BaseAgent):
    """
    Executes approved Linux/Unix shell commands with safety guardrails.
    Description: 'Execute system shell commands, file operations, and terminal tasks on the local machine.'
    """

    def __init__(self, llm_engine: LLMEngine, config: Optional[Union[Config, Dict[str, Any]]] = None):
        super().__init__(llm_engine, config=config)

    @property
    def name(self) -> str:
        return "system_command"

    @property
    def description(self) -> str:
        return "Execute system shell commands, file operations, and terminal tasks on the local machine."

    async def run(self, ctx: Context) -> AgentResult:
        """Execute shell commands with safety checks and confirmation."""
        # 1. Extract command from natural language using LLM
        prompt = f"""
Extract the single Linux shell command intended by this user query.
Query: {ctx.user_query}

Respond ONLY with the command string. No markdown, no explanations.
If no command is found, respond with 'none'.
"""
        try:
            res = await self.llm.chat([Message(role="user", content=prompt)])
            command = res.content.strip().strip('`').strip()
            
            if command.lower() == 'none' or not command:
                return AgentResult(content="I couldn't identify a specific shell command to run.", success=False)
        except Exception as e:
            logger.error(f"Failed to extract command: {e}")
            return AgentResult(content=f"Error extracting command: {str(e)}", success=False)

        return await self.execute_command(command)

    async def execute_command(self, command: str) -> AgentResult:
        """Validate, confirm and execute a specific shell command."""
        # 2. Safety Validation
        is_safe, msg = validate_shell_command(command, self.config)
        if not is_safe:
            logger.warning(f"Security block for command: {command} - Reason: {msg}")
            return AgentResult(content=f"Safety Block: {msg}", success=False)

        # 3. User Confirmation
        # Default to current workspace or home
        cwd_path = self.config.get("base_dir") if hasattr(self.config, "get") else None
        cwd = Path(cwd_path) if cwd_path else Path.cwd()
        timeout = self.config.get("security.shell_command_timeout", 30) if hasattr(self.config, "get") else 30

        console.print(Panel(
            f"[bold red]{command}[/bold red]\n\n"
            f"[dim]Working directory: {cwd}[/dim]\n"
            f"[dim]Timeout: {timeout}s[/dim]",
            title="FRIDAY Shell Execution",
            subtitle="Security Confirmation Required",
            border_style="yellow"
        ))
        
        # Confirmation prompt
        if not Confirm.ask("Do you want to proceed with this command?", default=False):
            return AgentResult(content="Command execution cancelled by user.")

        # 4. Execute command
        logger.info(f"Executing shell command: {command} (cwd={cwd})")
        exit_code, stdout, stderr = await run_shell_command(command, cwd=cwd, timeout=timeout)
        
        # 5. Format results
        result_text = f"Command: `{command}`\nExit Code: {exit_code}\n"
        
        if stdout:
            if len(stdout) > 2000:
                stdout = stdout[:2000] + "\n... (truncated)"
            result_text += f"\nSTDOUT:\n```\n{stdout}\n```"
            
        if stderr:
            if len(stderr) > 2000:
                stderr = stderr[:2000] + "\n... (truncated)"
            result_text += f"\nSTDERR:\n```\n{stderr}\n```"
            
        return AgentResult(
            content=result_text,
            success=(exit_code == 0),
            metadata={
                "exit_code": exit_code,
                "command": command,
                "cwd": str(cwd),
                "tts_content": f"Command executed with exit code {exit_code}."
            }
        )
