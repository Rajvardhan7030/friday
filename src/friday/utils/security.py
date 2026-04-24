"""Sandboxing helpers for secure code execution."""

import os
import subprocess
import tempfile
import logging
import sys
import shutil
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


def run_sandboxed_code(
    code: str, 
    workspace_dir: Path, 
    config: Optional[Config] = None
) -> Tuple[bool, str]:
    """Execute Python code in a restricted subprocess with multi-platform support.

    Args:
        code (str): Python source code to execute.
        workspace_dir (Path): Directory where execution happens.
        config (Config, optional): Central configuration object.

    Returns:
        Tuple[bool, str]: (success, output_or_error_message)
    """
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
                isolation_error = _validate_unshare_support()
                if isolation_error:
                    logger.error(isolation_error)
                    return False, isolation_error
                cmd = ["unshare", "-n"] + cmd

        # Execution flags
        kwargs: Dict[str, Any] = {
            "shell": False, 
            "cwd": str(workspace_dir), 
            "stdout": subprocess.PIPE, 
            "stderr": subprocess.PIPE, 
            "text": True
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
        else:
            # Windows specific flags
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(cmd, **kwargs)

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            success = process.returncode == 0
            output = stdout if success else stderr
            return success, output
            
        except subprocess.TimeoutExpired:
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


def _validate_unshare_support() -> Optional[str]:
    """Return an error message if unshare-based isolation cannot be used."""
    if not shutil.which("unshare"):
        return (
            "Error: Network-isolated sandbox requested but 'unshare' is not installed. "
            "Install it or change 'security.sandbox_backend'."
        )

    try:
        subprocess.run(["unshare", "-n", "true"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, PermissionError) as exc:
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
