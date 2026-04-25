"""
Sandbox Executor for FRIDAY Code Assistant.
Handles secure execution of Python code with import whitelisting and filesystem restrictions.
"""

import os
import sys
import ast
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional
from ..core.config import Config

logger = logging.getLogger(__name__)

# Security Hardening
ALLOWED_IMPORTS = {
    'os', 'pathlib', 'json', 'csv', 'datetime', 're', 
    'math', 'random', 'string', 'shutil', 'collections', 'itertools'
}

FORBIDDEN_PATTERNS = [
    'os.system', 'subprocess.', 'socket.', 'requests.', 'urllib.', 'shutil.rmtree'
]

class SandboxExecutor:
    """
    Executes Python code in a restricted environment.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.sandbox_dir = Path.home() / ".friday" / "sandbox"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = 10 # Hardcoded requirement

    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Performs static analysis to check for syntax errors and forbidden imports.
        """
        try:
            tree = ast.parse(code)
            
            # Check for forbidden imports/calls
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module]
                    for name in names:
                        if name and name.split('.')[0] not in ALLOWED_IMPORTS:
                            return False, f"Forbidden import: {name}. Only {ALLOWED_IMPORTS} are allowed."
                
                # Check calls for forbidden patterns
                if isinstance(node, ast.Call):
                    func_name = self._get_func_name(node.func)
                    for pattern in FORBIDDEN_PATTERNS:
                        if func_name and pattern in func_name:
                            return False, f"Forbidden function call: {func_name}"

            return True, "Syntax and safety check passed."
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"
        except Exception as e:
            return False, f"Validation Error: {str(e)}"

    def execute(self, code: str) -> Tuple[bool, str]:
        """
        Runs code in the sandbox directory with resource limits.
        """
        # Create temp script
        with tempfile.NamedTemporaryFile(suffix=".py", dir=self.sandbox_dir, mode='w', delete=False) as f:
            f.write(code)
            script_path = Path(f.name)

        try:
            # Use unshare for network isolation if on Linux
            cmd = [sys.executable, str(script_path)]
            if sys.platform != "win32":
                cmd = ["unshare", "-n"] + cmd

            result = subprocess.run(
                cmd,
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output
            
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {self.timeout}s."
        except Exception as e:
            return False, f"Execution failed: {str(e)}"
        finally:
            if script_path.exists():
                script_path.unlink()

    def _get_func_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._get_func_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return None
