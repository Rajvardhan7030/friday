"""Sandboxing helpers for secure code execution."""

import os
import subprocess
import tempfile
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def run_sandboxed_code(
    code: str, 
    workspace_dir: Path, 
    timeout: int = 30
) -> Tuple[bool, str]:
    """Execute Python code in a restricted subprocess."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file for the script
    with tempfile.NamedTemporaryFile(suffix=".py", dir=workspace_dir, mode='w', delete=False) as f:
        f.write(code)
        script_path = f.name

    process = None
    try:
        # Use current interpreter for consistency
        cmd = [sys.executable, script_path]
        
        # On Linux, try to run without network
        if os.name == "posix" and subprocess.call(["which", "unshare"], stdout=subprocess.DEVNULL) == 0:
            # unshare -n: run in new network namespace with only loopback
            cmd = ["unshare", "-n"] + cmd

        # Start process with its own session/process group to ensure it can't ignore signals
        kwargs: Dict[str, Any] = {"shell": False, "cwd": workspace_dir, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
        if os.name == "posix":
            kwargs["start_new_session"] = True

        process = subprocess.Popen(cmd, **kwargs)

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            success = process.returncode == 0
            output = stdout if success else stderr
            return success, output
        except subprocess.TimeoutExpired:
            # FORCE KILL the entire process group
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            return False, f"Error: Code execution timed out after {timeout} seconds."

    except Exception as e:
        logger.error(f"Sandboxed execution failed: {e}")
        return False, f"Error: Execution failed: {str(e)}"
    finally:
        # Cleanup script
        if os.path.exists(script_path):
            os.remove(script_path)
