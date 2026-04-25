"""
AutoGen Multi-Agent Coding Team for FRIDAY.
Orchestrates Coder, Executor, and Debugger agents for complex tasks.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message
from .human_proxy import HumanProxy
from .sandbox_executor import SandboxExecutor

logger = logging.getLogger(__name__)

class CodingTeam(BaseAgent):
    """
    A multi-agent team that collaborates to build complex software entirely locally.
    """

    def __init__(self, llm_engine: LLMEngine, config_list: List[Dict[str, Any]]):
        super().__init__(llm_engine)
        self.config_list = config_list
        self.proxy = HumanProxy(sandbox_dir="~/.friday/sandbox")
        self.executor = SandboxExecutor()

    @property
    def name(self) -> str: return "autogen_team"

    @property
    def description(self) -> str:
        return "Collaborative multi-agent team for building complete applications and complex systems."

    async def run(self, ctx: Context) -> AgentResult:
        """
        Simulates an AutoGen group chat workflow.
        """
        logger.info(f"Starting AutoGen Team for task: {ctx.user_query}")
        
        # Initial code generation
        coder_res = await self._step_coder(ctx.user_query)
        print(f"🧑‍💻 [Coder]: I've drafted the initial implementation.")
        
        history = [coder_res]
        success = False
        final_output = ""
        
        for turn in range(5): # Max 5 turns
            # 1. Executor runs the code
            print(f"⚙️ [Executor]: Validating and running code in sandbox...")
            valid, msg = self.executor.validate_syntax(coder_res)
            if not valid:
                exec_success, exec_output = False, msg
            else:
                exec_success, exec_output = self.executor.execute(coder_res)
            
            if exec_success:
                print(f"✅ [Executor]: Code executed successfully.")
                success = True
                final_output = exec_output
                break
            
            # 2. Debugger analyzes and suggests fixes
            print(f"🐛 [Debugger]: Execution failed. Analyzing logs...")
            debug_res = await self._step_debugger(coder_res, exec_output)
            
            # 3. Coder applies fixes
            print(f"🧑‍💻 [Coder]: Applying fixes suggested by Debugger...")
            coder_res = await self._step_coder(f"Fix this code based on error: {exec_output}. Suggestions: {debug_res}\nOriginal Code:\n{coder_res}")
            history.append(coder_res)

        return AgentResult(
            content=f"Team finished with {'success' if success else 'failure'}.\nFinal Output:\n{final_output}",
            metadata={"code": coder_res, "success": success}
        )

    async def _step_coder(self, prompt: str) -> str:
        system = "You are an expert Python developer. Write clean, modular, and safe code. Only use allowed libraries."
        res = await self.llm.chat([Message(role="system", content=system), Message(role="user", content=prompt)])
        return self._extract_code(res.content)

    async def _step_debugger(self, code: str, error: str) -> str:
        system = "You are a debugging specialist. Analyze the code and the error, and provide specific instructions on how to fix it."
        prompt = f"Code:\n{code}\n\nError:\n{error}"
        res = await self.llm.chat([Message(role="system", content=system), Message(role="user", content=prompt)])
        return res.content

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```python\s+(.*?)\s+```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: look for any code block if python tag is missing
        match = re.search(r"```\s+(.*?)\s+```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()
