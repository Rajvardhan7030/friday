"""
Resilient Code Assistant Agent for FRIDAY.
Implements a LangGraph-inspired state machine with planning, syntax validation, and sandboxed execution.
"""

import logging
import re
from typing import Dict, Any, List, Optional, TypedDict
from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message
from .sandbox_executor import SandboxExecutor

logger = logging.getLogger(__name__)

class CodeState(TypedDict):
    """State for the Code Assistant workflow."""
    task: str
    plan: str
    code: str
    success: bool
    output: str
    retries: int
    max_retries: int
    error: Optional[str]
    history: List[Dict[str, str]]

class CodeAssistantAgent(BaseAgent):
    """
    A robust coding agent that plans, writes, tests, and debugs code locally.
    """

    def __init__(self, llm_engine: LLMEngine, executor: Optional[SandboxExecutor] = None):
        super().__init__(llm_engine)
        self.executor = executor or SandboxExecutor()

    @property
    def name(self) -> str: return "code_assistant"

    @property
    def description(self) -> str:
        return "Expert Python developer that writes and executes code to solve tasks in a secure sandbox."

    async def run(self, ctx: Context) -> AgentResult:
        """Execute the multi-step coding workflow."""
        state: CodeState = {
            "task": ctx.user_query,
            "plan": "",
            "code": "",
            "success": False,
            "output": "",
            "retries": 0,
            "max_retries": 3,
            "error": None,
            "history": ctx.chat_history or []
        }

        # Step 1: Plan
        state = await self._plan(state)
        
        while state["retries"] < state["max_retries"]:
            # Step 2: Generate
            state = await self._generate_code(state)
            
            # Step 3: Validate Syntax (Static Analysis)
            valid, msg = self.executor.validate_syntax(state["code"])
            if not valid:
                state["error"] = f"Syntax/Safety Error: {msg}"
                state["retries"] += 1
                continue

            # Step 4: Execute in Sandbox
            success, output = await self.executor.execute(state["code"])
            state["success"] = success
            state["output"] = output

            # Step 5: Analyze Output
            if success:
                break
            else:
                state = await self._debug(state)
                state["retries"] += 1

        return self._format_result(state)

    async def _plan(self, state: CodeState) -> CodeState:
        messages = [
            Message(role=m["role"], content=m["content"]) 
            for m in state["history"][-6:]
        ]
        messages.append(Message(
            role="user", 
            content=f"Break this task into steps for a Python script: {state['task']}\nRespond with pseudocode steps."
        ))
        
        res = await self.llm.chat(messages)
        state["plan"] = res.content
        return state

    async def _generate_code(self, state: CodeState) -> CodeState:
        messages = [
            Message(role=m["role"], content=m["content"]) 
            for m in state["history"][-6:]
        ]
        
        prompt = f"""Write a Python script based on this plan:
{state['plan']}

Constraints:
- Only use allowed imports: os, pathlib, json, csv, datetime, re, math, random, string, shutil
- Files can only be written to current directory (sandbox).
- Task: {state['task']}

Respond ONLY with the code wrapped in ```python blocks."""
        
        if state["error"]:
            prompt += f"\n\nPrevious Error to fix: {state['error']}\nPrevious Code:\n{state['code']}"

        messages.append(Message(role="user", content=prompt))
        
        res = await self.llm.chat(messages)
        
        # Robust code extraction
        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", res.content, re.DOTALL | re.IGNORECASE)
        if code_match:
            state["code"] = code_match.group(1).strip()
        else:
            # Fallback: if no code block, try to find lines that look like code or just take the whole thing
            # but strip common conversational filler if it's very short
            state["code"] = res.content.strip()
            
        return state

    async def _debug(self, state: CodeState) -> CodeState:
        state["error"] = f"Execution failed with output:\n{state['output']}"
        logger.warning(f"Debugging attempt {state['retries'] + 1} for task: {state['task']}")
        return state

    def _format_result(self, state: CodeState) -> AgentResult:
        if state["success"]:
            voice_summary = f"I've successfully created and executed the script for {state['task']}."
            return AgentResult(
                content=f"Code executed successfully.\n\nOutput:\n{state['output']}\n\nCode:\n```python\n{state['code']}\n```",
                metadata={"tts_content": voice_summary, "success": True}
            )
        else:
            error_summary = f"I tried 3 times but couldn't fix the code. The final error was: {state['output'][:100]}"
            return AgentResult(
                content=f"Failed to complete task after {state['max_retries']} attempts.\nLast Output:\n{state['output']}",
                success=False,
                metadata={"tts_content": error_summary}
            )
