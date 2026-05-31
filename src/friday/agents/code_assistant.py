"""
Resilient Code Assistant Agent for FRIDAY.
Implements a LangGraph-inspired state machine with planning, syntax validation, and sandboxed execution.
"""

import logging
import re
from typing import Dict, Any, List, Optional, TypedDict, Union

from .base import BaseAgent, Context, AgentResult, AgentMetadata
from ..llm.engine import LLMEngine, Message
from .sandbox_executor import SandboxExecutor

logger = logging.getLogger(__name__)

class CodeState(TypedDict):
    """Internal state for the code assistant workflow."""
    task: str
    plan: Optional[str]
    code: Optional[str]
    output: Optional[str]
    error: Optional[str]
    retries: int
    max_retries: int
    success: bool

class CodeAssistantAgent(BaseAgent):
    """
    Autonomous coding agent that plans, writes, and executes Python code in a sandbox.
    """

    def __init__(
        self, 
        llm_engine: Union[LLMEngine, 'ModelRouter'], 
        sandbox: SandboxExecutor,
        max_retries: int = 3,
        config: Optional[Any] = None
    ):
        super().__init__(llm_engine, config=config)
        self.sandbox = sandbox
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        return "code_assistant"

    @property
    def description(self) -> str:
        return "Autonomous Python coder that creates and runs scripts in a secure sandbox."

    async def run(self, ctx: Context) -> AgentResult:
        """Execute the coding workflow."""
        state: CodeState = {
            "task": ctx.user_query,
            "plan": None,
            "code": None,
            "output": None,
            "error": None,
            "retries": 0,
            "max_retries": self.max_retries,
            "success": False
        }

        # Workflow Loop
        while state["retries"] < state["max_retries"]:
            # 1. Plan
            if not state["plan"]:
                state = await self._plan(state)
            
            # 2. Generate Code
            state = await self._generate_code(state)
            
            # 3. Execute in Sandbox
            state = await self._execute(state)
            
            if state["success"]:
                break
            else:
                # 4. Debug/Retry
                state = await self._debug(state)
                state["retries"] += 1

        return self._format_result(state)

    async def _plan(self, state: CodeState) -> CodeState:
        """Creates a step-by-step plan for the coding task."""
        prompt = f"Create a step-by-step technical plan to solve this task: {state['task']}"
        llm = await self._get_llm()
        res = await llm.chat([Message(role="user", content=prompt)])
        state["plan"] = res.content
        logger.info(f"Plan generated for task: {state['task']}")
        return state

    async def _generate_code(self, state: CodeState) -> CodeState:
        """Writes the Python code based on the plan and any previous errors."""
        error_context = f"\nPrevious Error: {state['error']}" if state["error"] else ""
        prompt = f"""
Write a complete, single-file Python script to solve this task: {state['task']}
Plan: {state['plan']}
{error_context}

Guidelines:
- Use standard libraries or those available in the environment.
- Print the final result clearly to stdout.
- Return ONLY the Python code inside triple backticks.
"""
        llm = await self._get_llm()
        res = await llm.chat([Message(role="user", content=prompt)])
        
        # Extract code from markdown
        code_match = re.search(r"```python\n(.*?)\n```", res.content, re.DOTALL)
        if code_match:
            state["code"] = code_match.group(1)
        else:
            state["code"] = res.content # Fallback if no backticks
            
        return state

    async def _execute(self, state: CodeState) -> CodeState:
        """Runs the code in the sandboxed environment."""
        if not state["code"]:
            state["error"] = "No code generated."
            return state

        success, output = await self.sandbox.execute(state["code"])
        state["output"] = output
        state["success"] = success
        
        if not success:
            state["error"] = output
            
        return state

    async def _debug(self, state: CodeState) -> CodeState:
        """Analyzes the error and prepares for the next attempt."""
        state["error"] = f"Execution failed with output:\n{state['output']}"
        logger.warning(f"Debugging attempt {state['retries'] + 1} for task: {state['task']}")
        return state

    def _format_result(self, state: CodeState) -> AgentResult:
        from .base import AgentMetadata
        if state["success"]:
            voice_summary = f"I've successfully created and executed the script for {state['task']}."
            return AgentResult(
                content=f"Code executed successfully.\n\nOutput:\n{state['output']}\n\nCode:\n```python\n{state['code']}\n```",
                metadata=AgentMetadata(tts_content=voice_summary),
                success=True
            )
        else:
            error_summary = f"I tried 3 times but couldn't fix the code. The final error was: {state['output'][:100]}"
            return AgentResult(
                content=f"Failed to complete task after {state['max_retries']} attempts.\\nLast Output:\\n{state['output']}",
                success=False,
                metadata=AgentMetadata(tts_content=error_summary)
            )
