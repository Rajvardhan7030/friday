"""
Markdown Validator Agent for FRIDAY.
Ensures generated documentation is well-formed and consistent.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message

logger = logging.getLogger(__name__)

class MarkdownValidator(BaseAgent):
    """
    Validates markdown files for hierarchy, links, and style.
    """

    def __init__(self, llm_engine: LLMEngine, fix: bool = False):
        super().__init__(llm_engine)
        self.fix = fix

    @property
    def name(self) -> str: return "md_validator"

    @property
    def description(self) -> str:
        return "Validates and auto-fixes markdown files for consistency and correctness."

    async def run(self, ctx: Context) -> AgentResult:
        """
        Validates a markdown string or file.
        """
        content = ctx.user_query
        file_path = None
        
        # Check if query is a path
        if len(content) < 255 and Path(content).exists():
            file_path = Path(content)
            with open(file_path, 'r') as f:
                content = f.read()

        issues = self._check_rules(content)
        
        fixed_content = content
        if issues and self.fix:
            print(f"🛠️ [Validator]: Auto-fixing {len(issues)} issues...")
            fixed_content = await self._auto_fix(content, issues)
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(fixed_content)

        return AgentResult(
            content=fixed_content,
            success=len(issues) == 0 or self.fix,
            metadata={
                "issues": issues,
                "was_fixed": self.fix and len(issues) > 0
            }
        )

    def _check_rules(self, content: str) -> List[str]:
        issues = []
        lines = content.split('\n')
        
        # 1. Heading Hierarchy
        headings = [len(m.group(1)) for m in re.finditer(r'^(#+)\s', content, re.M)]
        for i in range(1, len(headings)):
            if headings[i] > headings[i-1] + 1:
                issues.append(f"Heading skip: H{headings[i]} follows H{headings[i-1]}")

        # 2. Long Lines
        for i, line in enumerate(lines):
            if len(line) > 120:
                issues.append(f"Line {i+1} is too long ({len(line)} chars)")

        # 3. Empty code blocks
        if "```\n```" in content:
            issues.append("Empty code block detected")

        return issues

    async def _auto_fix(self, content: str, issues: List[str]) -> str:
        prompt = f"Fix the following markdown content based on these issues: {issues}\n\nContent:\n{content}"
        res = await self.llm.chat([Message(role="user", content=prompt)])
        return res.content
