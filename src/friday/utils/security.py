"""Sandboxing helpers for secure code execution."""

import os
import tempfile
import logging
import sys
import shutil
import ast
import asyncio
import re
import signal
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
    '__func__', '__self__', '__dict__', '__class__', '__mro__',
    '__getattribute__', '__getattr__', '__setattr__', '__delattr__'
}

FORBIDDEN_BUILTINS = {
    'eval', 'exec', 'open', 'compile', 'input', 'globals', 'locals', 
    '__import__', 'getattr', 'setattr', 'delattr', 'hasattr', 'breakpoint'
}

FORBIDDEN_PATTERNS = [
    'os.system', 'os.remove', 'os.unlink', 'os.rename', 'os.replace', 'os.rmdir', 
    'os.removedirs', 'os.mkdir', 'os.makedirs', 'os.chmod', 'os.chown', 'os.lchown', 
    'os.symlink', 'os.link', 'os.chdir', 'os.fchdir', 'os.chroot',
    'subprocess.', 'socket.', 'requests.', 'urllib.', 'shutil.rmtree', 'shutil.copy', 
    'shutil.move',
    'read_text', 'read_bytes', 'write_text', 'write_bytes', 'unlink', 'rmdir', 
    'rename', 'replace', 'chmod', 'lchmod', 'symlink_to', 'hardlink_to', 'mkdir'
]

SHELL_BLOCKLIST = [
    "rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:", "> /dev/", 
    "chown", "chmod 777", "rm -rf /home", "rm -rf /etc", "rm -rf /root",
    "| sh", "| bash", "| zsh", "| ksh", "| dash",
    "python -c", "python3 -c", "perl -e", "ruby -e", "lua -e",
    "curl", "wget", "bash -i", "sh -i", "nc ", "netcat ", "/etc/shadow", "/etc/sudoers"
]

SYSTEM_DIRS = ["/bin", "/sbin", "/usr", "/etc", "/sys", "/proc", "/dev", "/root", "/var", "/lib"]

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
            
            # 3. Check names (prevents aliasing like e = eval)
            if isinstance(node, ast.Name):
                if node.id in FORBIDDEN_BUILTINS or node.id == "__builtins__":
                    return False, f"Forbidden name: {node.id}"

            # 4. Check Subscript access (prevents __builtins__['eval'])
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == "__builtins__":
                    return False, "Forbidden access to __builtins__ via subscript."

            # 5. Check calls for forbidden patterns
            if isinstance(node, ast.Call):
                func_name = _get_func_name(node.func)
                if func_name:
                    for pattern in FORBIDDEN_PATTERNS:
                        if pattern in func_name:
                            return False, f"Forbidden function call: {func_name}"

        return True, "Syntax and safety check passed."
    except SyntaxError as e:
        return False, f"Syntax Error: {str(e)}"
    except Exception as e:
        return False, f"Validation Error: {str(e)}"

ALLOWED_BINARIES = {
    "ls", "pwd", "date", "whoami", "uname", "hostname", "df", "du", "free", 
    "uptime", "mkdir", "touch", "cat", "grep", "head", "tail", "echo", "find",
    "ps", "history", "type", "which", "cp", "mv", "rm", "sleep"
}

def _has_unquoted_char(s: str, chars: set[str]) -> bool:
    """Helper to detect if any of the target characters exist outside of quotes."""
    in_single = False
    in_double = False
    escaped = False
    for char in s:
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if not in_single and not in_double and char in chars:
            return True
    return False

def validate_shell_command(command: str, config: Optional[Config] = None) -> Tuple[bool, str]:
    """
    Checks if a shell command is safe to execute based on allowlists, blocklists and structure.
    """
    if not command.strip():
        return False, "Empty command"

    # 0. Check for forbidden characters that enable chaining or substitution
    if "\n" in command or "\r" in command:
        return False, "Newlines are prohibited in shell commands."
    
    # Check for command/process substitution and environment variable expansion
    if any(x in command for x in ["$(", "`", "${"]):
        return False, "Command/process substitution or environment variable expansion ($(), ``, ${}) is prohibited."
    
    if "$" in command:
        return False, "Environment variable expansion ($) is prohibited."

    # Check for unquoted redirection operators or subshells
    if _has_unquoted_char(command, {">", "<"}):
        return False, "Redirection operators (> or <) are prohibited outside of quotes."

    if _has_unquoted_char(command, {"(", ")"}):
        return False, "Subshells or parentheses are prohibited outside of quotes." 

    # 1. Basic Sudo check
    cmd_lower = command.lower()
    allow_sudo = config.get("security.shell_command_allow_sudo", False) if config else False
    if "sudo" in cmd_lower and not allow_sudo:
        return False, "Sudo commands are disabled in configuration."

    import shlex
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return False, f"Invalid shell syntax: {str(e)}"

    if not tokens:
        return False, "Empty command"

    # 2. Structural Validation and Binary Allowlist
    # A command starts at the beginning or after a separator
    separators = {";", "&&", "||", "|"}
    expect_binary = True
    in_sudo_prefix = False
    skip_next_option_arg = False
    sudo_opts_with_arg = {"-u", "-g", "-C", "-h", "-p", "-r", "-t", "-U"}
    
    for i, token in enumerate(tokens):
        if expect_binary:
            if token == "sudo" and allow_sudo:
                in_sudo_prefix = True
                continue
            
            if in_sudo_prefix:
                if skip_next_option_arg:
                    skip_next_option_arg = False
                    continue
                if token in sudo_opts_with_arg:
                    skip_next_option_arg = True
                    continue
                if token.startswith("-"):
                    continue
            
            # Identify the binary token
            binary = token.split('/')[-1]
            if binary not in ALLOWED_BINARIES:
                return False, f"Forbidden or unknown binary detected: {binary}. Only {sorted(list(ALLOWED_BINARIES))} are allowed."
            
            expect_binary = False
            in_sudo_prefix = False
            skip_next_option_arg = False
            continue

        if token in separators:
            expect_binary = True
            in_sudo_prefix = False
            skip_next_option_arg = False
            continue
        
        # Check for forbidden redirection targets or patterns in arguments
        # (Though most of this is caught by system directory protection below)
        pass

    # 3. Blocklist and System Directory Protection (Legacy but good as extra layer)
    # We re-verify the full command against some hardcoded dangerous patterns
    for pattern in SHELL_BLOCKLIST:
        if pattern in command:
            return False, f"Dangerous command pattern detected: {pattern}"

    # 4. System Directory Protection
    for sdir in SYSTEM_DIRS:
        if sdir in command:
            # Modification check
            if any(x in command for x in ["rm", "mv", "cp", "touch", ">", ">>", "tee", "chmod", "chown"]):
                return False, f"Modification of system directory {sdir} is prohibited."
            # Sensitive access check
            if sdir == "/etc" and any(x in command for x in ["shadow", "sudoers", "passwd", "group", "gshadow"]):
                if any(x in command for x in ["cat", "less", "more", "head", "tail", "nano", "vim", "vi", "grep"]):
                    return False, f"Access to sensitive file in {sdir} is prohibited."

    # 5. Dangerous 'rm' targets
    if "rm " in command and ("-r" in command or "-f" in command):
        if any(x in command for x in [" /", " .", " ..", " *", " ~"]):
             return False, f"Dangerous 'rm' target detected."

    return True, "Command validated."

async def run_shell_command(
    command: str,
    cwd: Optional[Path] = None,
    timeout: int = 30
) -> Tuple[int, str, str]:
    """
    Executes a shell command safely. 
    Uses create_subprocess_shell to support pipes and redirections.
    
    Returns:
        Tuple[int, str, str]: (exit_code, stdout, stderr)
    """
    try:
        if not command.strip():
            return -1, "", "Empty command"

        # Use start_new_session=True on POSIX to enable reliable process group killing
        kwargs = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": str(cwd) if cwd else None
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True

        # We use create_subprocess_shell to support shell features like pipes and redirects.
        # Security is handled by validate_shell_command() which MUST be called before this.
        process = await asyncio.create_subprocess_shell(
            command,
            **kwargs
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return process.returncode, stdout.decode(errors='replace'), stderr.decode(errors='replace')
        except TimeoutError:
            _kill_process_tree(process.pid)
            return -1, "", f"Command timed out after {timeout} seconds."
            
    except Exception as e:
        logger.error(f"Shell execution failed: {e}")
        return -1, "", f"Error: {str(e)}"

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
                # Improved isolation: 
                # -r: map current user to root (user namespace)
                # -n: network namespace (no network)
                # -m: mount namespace (filesystem isolation)
                # -p -f: PID namespace (cannot see other processes)
                # --mount-proc: mount a new /proc for the PID namespace
                cmd = ["unshare", "-rn", "-m", "-p", "-f", "--mount-proc"] + cmd

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
            
            # Fix: Combine stdout and stderr on failure so preceding stdout output is not lost
            if success:
                output_bytes = stdout
            else:
                output_bytes = stdout + b"\n" + stderr if stdout else stderr
                
            output = output_bytes.decode('utf-8', errors='replace')
                
            return success, output
            
        except TimeoutError:
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
            "Error: Sandbox requested but 'unshare' is not installed. "
            "Install it or change 'security.sandbox_backend'."
        )

    # We check for the most restrictive set of flags we use: -rnmpf --mount-proc
    # Some systems might not support all, so we might need a tiered check, 
    # but for security we should prefer failing if isolation is weak.
    try:
        proc = await asyncio.create_subprocess_exec(
            "unshare", "-rn", "-m", "-p", "-f", "--mount-proc", "true",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        if proc.returncode == 0:
            return None
    except Exception:
        pass

    return (
        "Error: Enhanced isolation (unshare -rnmpf) is not permitted on this system. "
        "Ensure user namespaces are enabled or choose a different sandbox backend."
    )


def _kill_process_tree(pid: int) -> None:
    """Force kill a process and all its children."""
    if not psutil:
        # Fallback if psutil not available
        try:
            if os.name == "posix":
                # On POSIX, kill the entire process group
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            # Fallback to single process kill if pgid fails
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
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
