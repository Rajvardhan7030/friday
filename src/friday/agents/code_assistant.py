"""Code Assistant Agent with CodeAct pattern."""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from friday.agents.base import BaseAgent, Context, AgentResult
from friday.llm.engine import LLMEngine, Message
from friday.utils.security import run_sandboxed_code

logger = logging.getLogger(__name__)

class CodeAssistantAgent(BaseAgent):
    """Agent that writes and executes Python/bash code to solve tasks."""

    def __init__(self, llm_engine: LLMEngine, workspace_dir: Path):
        super().__init__(llm_engine)
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return "Code assistant that can write and execute Python code in a sandbox."

    async def run(self, ctx: Context) -> AgentResult:
        """Run the CodeAct loop."""
        logger.info(f"Solving code task: {ctx.user_query}")
        
        # v0.1: single turn CodeAct for simplicity
        messages = [
            Message(role="system", content=f"You are Friday Code Assistant. You solve tasks by writing Python code. Wrap your code in ```python ... ``` blocks. You have access to a workspace directory at {self.workspace_dir}."),
            Message(role="user", content=ctx.user_query)
        ]
        
        response = await self.llm.chat(messages)
        content = response.content
        
        # Parse and execute code blocks
        code_blocks = self._extract_python_code(content)
        execution_results = []
        
        for code in code_blocks:
            success, output = run_sandboxed_code(code, self.workspace_dir)
            execution_results.append({
                "code": code,
                "success": success,
                "output": output
            })
            
        # Final synthesis
        if execution_results:
            synthesis_prompt = f"Original Query: {ctx.user_query}\n\nCode execution results:\n"
            for res in execution_results:
                synthesis_prompt += f"--- CODE ---\n{res['code']}\n--- OUTPUT ---\n{res['output']}\n\n"
            synthesis_prompt += "Please summarize the results above."
            
            synthesis_response = await self.llm.chat([
                Message(role="system", content="Summarize the code execution results clearly for the user."),
                Message(role="user", content=synthesis_prompt)
            ])
            content = synthesis_response.content

        return AgentResult(
            content=content,
            metadata={"executions": execution_results}
        )

    def _extract_python_code(self, text: str) -> List[str]:
        """Extract code from markdown python blocks."""
        pattern = r"```python\n(.*?)\n```"
        return re.findall(pattern, text, re.DOTALL)
