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
import io
import contextlib
from pathlib import Path
from typing import Tuple, List, Optional
from friday.core.config import Config

try:
    import docker
except ImportError:
    docker = None

logger = logging.getLogger(__name__)

# Security Hardening
ALLOWED_IMPORTS = {
    'os', 'pathlib', 'json', 'csv', 'datetime', 're', 
    'math', 'random', 'string', 'shutil', 'collections', 'itertools'
}

FORBIDDEN_PATTERNS = [
    'os.system', 'subprocess.', 'socket.', 'requests.', 'urllib.', 'shutil.rmtree', 'eval', 'exec', 'open'
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
        
        self.docker_client = None
        if docker:
            try:
                self.docker_client = docker.from_env()
                self.docker_client.ping()
            except Exception as e:
                logger.warning(f"Docker client initialization failed: {e}. Will fallback to restricted local execution.")
                self.docker_client = None

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
        Attempts Docker first, falls back to restricted Python execution.
        """
        # Create temp script
        script_path = None
        with tempfile.NamedTemporaryFile(suffix=".py", dir=self.sandbox_dir, mode='w', delete=False) as f:
            f.write(code)
            script_path = Path(f.name)

        try:
            if self.docker_client:
                return self._execute_docker(script_path)
            else:
                return self._execute_restricted_local(code)
        finally:
            if script_path and script_path.exists():
                script_path.unlink()

    def _execute_docker(self, script_path: Path) -> Tuple[bool, str]:
        """Execute script inside an ephemeral docker container."""
        try:
            container = self.docker_client.containers.run(
                "python:3.11-slim",
                command=["python", f"/workspace/{script_path.name}"],
                volumes={str(self.sandbox_dir): {'bind': '/workspace', 'mode': 'ro'}},
                working_dir="/workspace",
                network_mode="none",
                mem_limit="128m",
                cpu_quota=50000,
                detach=True
            )
            
            try:
                result = container.wait(timeout=self.timeout)
                success = result.get("StatusCode", 1) == 0
            except Exception as e: # Catch wait timeout
                container.kill()
                return False, f"Execution timed out after {self.timeout}s."
                
            logs = container.logs().decode("utf-8")
            container.remove()
            return success, logs
            
        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return False, f"Execution failed: {str(e)}"

    def _execute_restricted_local(self, code: str) -> Tuple[bool, str]:
        """Safe local fallback using restricted execution."""
        logger.warning("Using local fallback execution environment.")
        
        output_capture = io.StringIO()
        error_capture = io.StringIO()
        
        # Define safe globals
        safe_globals = {
            "__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "bool": bool,
                "abs": abs,
                "sum": sum,
                "min": min,
                "max": max,
                "round": round,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "__import__": __import__,  # Needed to allow the whitelisted imports
            }
        }

        success = False
        try:
            # We redirect stdout and stderr
            with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(error_capture):
                # The syntax validator already checks AST for dangerous calls, 
                # so this provides a second layer of defense.
                exec(code, safe_globals, {})
            success = True
            output = output_capture.getvalue()
        except Exception as e:
            output = f"Restricted execution error: {type(e).__name__}: {str(e)}"
        
        return success, output

    def _get_func_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._get_func_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return None