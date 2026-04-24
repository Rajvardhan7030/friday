"""Code Assistant Agent that registers as a command handler."""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..core.registry import registry
from ..core.agent_runner import Session
from ..core.config import Config
from ..llm.engine import LLMEngine, Message
from ..utils.security import run_sandboxed_code

logger = logging.getLogger(__name__)


def _resolve_workspace_dir(
    session: Session,
    task_description: str,
    config: Optional[Config] = None,
) -> Path:
    """Choose the workspace directory for code generation tasks."""
    configured_base_dir = Path(getattr(config, "base_dir", Config.DEFAULT_BASE_DIR))
    workspace_dir = configured_base_dir / "workspace"

    previous_message = session.history[-1]["content"].lower() if session.history else ""
    if "desktop" in task_description.lower() or "desktop" in previous_message:
        return Path.home() / "Desktop"

    return workspace_dir

@registry.register(
    name="Code Task",
    regex=r"(?:create|write|run|execute|make|generate) (?:a )?(?:file|code|script|program) (.+)",
    description="Handle coding and file system tasks",
    usage="create a file named hello.py",
    priority=5
)
async def code_task_handler(session: Session, task_description: str, llm: Optional[LLMEngine] = None, config: Optional[Config] = None):
    """
    Handler for coding tasks that uses the LLM to generate code 
    and executes it in a sandbox.
    """
    if not llm:
        return "Internal Error: LLM engine not available for coding tasks."
    
    workspace_dir = _resolve_workspace_dir(session, task_description, config)

    logger.info(f"Handling code task: {task_description}")
    
    # Construct the prompt for the Code Assistant
    messages = [
        Message(role="system", content=(
            "You are Friday Code Assistant. You solve tasks by writing Python code. "
            "Wrap your code in ```python ... ``` blocks. "
            f"Current workspace directory is: {workspace_dir}. "
            "Focus on the request and write clean, safe code."
        )),
        Message(role="user", content=task_description)
    ]
    
    try:
        response = await llm.chat(messages)
        content = response.content
        
        # Parse and execute code blocks
        code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
        execution_results = []
        
        for code in code_blocks:
            success, output = run_sandboxed_code(code, workspace_dir, config=config)
            execution_results.append({
                "code": code,
                "success": success,
                "output": output
            })
            
        # Final synthesis
        if execution_results:
            synthesis_prompt = f"Original Task: {task_description}\n\nCode execution results:\n"
            for res in execution_results:
                synthesis_prompt += f"--- CODE ---\n{res['code']}\n--- OUTPUT ---\n{res['output']}\n\n"
            synthesis_prompt += "Please summarize if the task was completed successfully based on the output."
            
            synthesis_response = await llm.chat([
                Message(role="system", content="Summarize the code execution results clearly for the user."),
                Message(role="user", content=synthesis_prompt)
            ])
            return synthesis_response.content

        return content # Return original response if no code blocks found
        
    except Exception as e:
        logger.error(f"Code assistant handler failed: {e}")
        return f"I tried to solve that task with code, but something went wrong: {str(e)}"
