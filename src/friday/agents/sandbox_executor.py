"""
Sandbox Executor for FRIDAY Code Assistant.
Handles secure execution of Python code with import whitelisting and filesystem restrictions.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
from friday.core.config import Config
from friday.utils.security import run_sandboxed_code, validate_python_code

logger = logging.getLogger(__name__)

class SandboxExecutor:
    """
    Executes Python code in a restricted environment.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.sandbox_dir = self.config.base_dir / "sandbox"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Performs static analysis to check for syntax errors and forbidden imports.
        """
        return validate_python_code(code)

    def execute(self, code: str) -> Tuple[bool, str]:
        """
        Runs code in the sandbox directory with resource limits.
        Delegates to centralized security utility.
        """
        return run_sandboxed_code(code, self.sandbox_dir, self.config)
