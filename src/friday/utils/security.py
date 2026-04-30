"""Sandboxing helpers for secure code execution."""

import os
import tempfile
import logging
import sys
import shutil
import ast
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

try:
    import psutil
except ImportError:
    psutil = None

# resource module only exists on Unix
try:
    import resource
except ImportError:
    resource = None

from ..core.config import Config
from ..core.exceptions import SandboxError

logger = logging.getLogger(__name__)

# Security Hardening
ALLOWED_IMPORTS = {
    'os', 'pathlib', 'json', 'csv', 'datetime', 're', 
    'math', 'random', 'string', 'shutil', 'collections', 'itertools'
}

FORBIDDEN_ATTRIBUTES = {
    '__globals__', '__subclasses__', '__builtins__', '__code__', 
    '__func__', '__self__', '__dict__', '__class__', '__mro__'
}

FORBIDDEN_PATTERNS = [
    'os.system', 'subprocess.', 'socket.', 'requests.', 'urllib.', 'shutil.rmtree', 'eval', 'exec', 'open'
]

def validate_python_code(code: str) -> Tuple[bool, str]:
    """
    Performs robust static analysis to check for syntax errors and forbidden imports/calls/attributes.
    """
    try:
        tree = ast.parse(code)
        
        # Check for forbidden imports/calls/attributes
        for node in ast.walk(tree):
            # 1. Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module]
                for name in names:
                    if not name: continue
                    root_module = name.split('.')[0]
                    if root_module not in ALLOWED_IMPORTS:
                        return False, f"Forbidden import: {name}. Only {ALLOWED_IMPORTS} are allowed."
            
            # 2. Check attribute access (prevents bypasses like os.__dict__['system'])
            if isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTRIBUTES:
                    return False, f"Forbidden attribute access: {node.attr}"
            
            # 3. Check calls for forbidden patterns
            if isinstance(node, ast.Call):
                func_name = _get_func_name(node.func)
                if func_name:
                    for pattern in FORBIDDEN_PATTERNS:
                        if pattern in func_name:
                            return False, f"Forbidden function call: {func_name}"
                
                # Check for dynamic attribute access via getattr/setattr
                if isinstance(node.func, ast.Name) and node.func.id in ('getattr', 'setattr', 'delattr', 'hasattr'):
                    return False, f"Forbidden built-in call: {node.func.id}"

        return True, "Syntax and safety check passed."
    except SyntaxError as e:
        return False, f"Syntax Error: {str(e)}"
    except Exception as e:
        return False, f"Validation Error: {str(e)}"

def _get_func_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        val = _get_func_name(node.value)
        return f"{val}.{node.attr}" if val else node.attr
    return None

async def run_sandboxed_code(
    code: str, 
    workspace_dir: Path, 
    config: Optional[Config] = None
) -> Tuple[bool, str]:
    """Execute Python code in a restricted subprocess with multi-platform support.
    This function is asynchronous and does not block the event loop.

    Args:
        code (str): Python source code to execute.
        workspace_dir (Path): Directory where execution happens.
        config (Config, optional): Central configuration object.

    Returns:
        Tuple[bool, str]: (success, output_or_error_message)
    """
    # 1. Static Validation
    valid, msg = validate_python_code(code)
    if not valid:
        return False, msg

    timeout = config.get("security.sandbox_timeout", 30) if config else 30
    backend = config.get("security.sandbox_backend", "unshare") if config else "unshare"
    
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file for the script
    with tempfile.NamedTemporaryFile(suffix=".py", dir=workspace_dir, mode='w', delete=False) as f:
        f.write(code)
        script_path = Path(f.name)

    process = None
    try:
        # Use current interpreter for consistency
        cmd = [sys.executable, str(script_path)]
        
        # Isolation strategy based on OS and backend
        if os.name == "posix":
            if backend == "docker" and shutil.which("docker"):
                # Docker backend: maximum isolation
                cmd = [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "--memory", "128m",
                    "--cpus", "0.5",
                    "-v", f"{workspace_dir}:/workspace:ro",
                    "python:3.11-slim",
                    "python", f"/workspace/{script_path.name}"
                ]
            elif backend == "docker":
                return False, (
                    "Error: Docker sandbox requested but Docker is not available. "
                    "Install Docker or change 'security.sandbox_backend'."
                )
            elif backend == "unshare":
                isolation_error = await _validate_unshare_support()
                if isolation_error:
                    return False, isolation_error
                cmd = ["unshare", "-n"] + cmd

        # Execution flags
        kwargs: Dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE, 
            "stderr": asyncio.subprocess.PIPE, 
        }
        
        if os.name == "posix":
            kwargs["start_new_session"] = True
            # Set resource limits on Linux
            if resource:
                def preexec():
                    # Limit CPU time (seconds)
                    resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 5))
                    # Limit memory (128MB)
                    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
                kwargs["preexec_fn"] = preexec

        process = await asyncio.create_subprocess_exec(*cmd, **kwargs)

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            success = process.returncode == 0
            
            output_bytes = stdout if success else stderr
            output = output_bytes.decode('utf-8', errors='replace')
                
            return success, output
            
        except asyncio.TimeoutExpired:
            _kill_process_tree(process.pid)
            return False, f"Error: Code execution timed out after {timeout} seconds."

    except Exception as e:
        logger.error(f"Sandboxed execution failed: {e}")
        return False, f"Error: Execution failed: {str(e)}"
    finally:
        # Cleanup script
        if script_path.exists():
            try:
                script_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup script {script_path}: {e}")


async def _validate_unshare_support() -> Optional[str]:
    """Return an error message if unshare-based isolation cannot be used."""
    if not shutil.which("unshare"):
        return (
            "Error: Network-isolated sandbox requested but 'unshare' is not installed. "
            "Install it or change 'security.sandbox_backend'."
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            "unshare", "-n", "true",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        if proc.returncode != 0:
             return "Error: Network-isolated sandbox requested but 'unshare -n' failed."
    except Exception as exc:
        return (
            "Error: Network-isolated sandbox requested but 'unshare -n' is not permitted "
            f"on this system: {exc}. Adjust permissions or choose a different sandbox backend."
        )

    return None


def _kill_process_tree(pid: int) -> None:
    """Force kill a process and all its children."""
    if not psutil:
        # Fallback if psutil not available
        try:
            os.kill(pid, 9)
        except:
            pass
        return

    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        pass
