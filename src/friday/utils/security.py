"""Sandboxing helpers for secure code execution."""

import os
import subprocess
import tempfile
import logging
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

    try:
        # Command for restricted execution. 
        # v0.1: Best-effort sandbox. 
        # Linux: Use unshare to disable network if available.
        # macOS/Windows: Standard subprocess.
        cmd = ["python3", script_path]
        
        # On Linux, try to run without network
        if os.name == "posix" and subprocess.call(["which", "unshare"], stdout=subprocess.DEVNULL) == 0:
            # unshare -n: run in new network namespace with only loopback
            cmd = ["unshare", "-n"] + cmd

        result = subprocess.run(
            cmd,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            # Ensure agent can't break out easily (e.g., shell=False)
            shell=False
        )

        success = result.returncode == 0
        output = result.stdout if success else result.stderr
        return success, output

    except subprocess.TimeoutExpired:
        return False, f"Error: Code execution timed out after {timeout} seconds."
    except Exception as e:
        logger.error(f"Sandboxed execution failed: {e}")
        return False, f"Error: Execution failed: {str(e)}"
    finally:
        # Cleanup script
        if os.path.exists(script_path):
            os.remove(script_path)
