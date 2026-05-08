"""Structured logging to local rotating files only."""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler


import sys
import contextlib


@contextlib.contextmanager
def ignore_stderr():
    """Context manager to silence C-level stderr (e.g., from noisy libraries like PortAudio).
    
    It redirects file descriptor 2 (stderr) to /dev/null for the duration of the context.
    Python's sys.stderr is flushed first to ensure no Python-level logs are lost.
    """
    if os.name != "posix":
        # Non-POSIX systems might not support dup2 or /dev/null
        yield
        return

    try:
        # 1. Flush Python-level stderr to avoid losing buffered logs
        sys.stderr.flush()
        
        # 2. Save original stderr file descriptor
        old_stderr_fd = os.dup(sys.stderr.fileno())
        
        # 3. Open /dev/null
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        
        try:
            # 4. Redirect stderr (FD 2) to /dev/null
            os.dup2(devnull_fd, sys.stderr.fileno())
            
            yield
        finally:
            # 5. Restore original stderr
            sys.stderr.flush()
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            
            # 6. Cleanup
            os.close(old_stderr_fd)
            os.close(devnull_fd)
            
    except (OSError, AttributeError) as e:
        # If any system call fails, just yield and hope for the best
        logger.debug(f"Failed to redirect stderr: {e}")
        yield


def setup_logging(
    log_file: Path,
    level: int = logging.INFO,
    rotate_daily: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Setup structured logging to a local rotating file and console.

    Args:
        log_file (Path): Path to the log file.
        level (int, optional): Logging level. Defaults to logging.INFO.
        rotate_daily (bool, optional): Whether to use TimedRotatingFileHandler. Defaults to True.
        max_bytes (int, optional): Maximum size of each log file if not rotating daily. Defaults to 10MB.
        backup_count (int, optional): Number of backup files to keep. Defaults to 5.
    """
    # Prevent duplicate handlers if setup_logging is called multiple times
    if logging.getLogger().handlers:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Use a structured format for the log file
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set up file handler
    if rotate_daily:
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=backup_count
        )
    else:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Ensure log file is not world-readable (0o600)
    # We do this after creation since open() might create it.
    if not log_file.exists():
        log_file.touch(mode=0o600)
    else:
        os.chmod(log_file, 0o600)

    # Console handler using Rich for pretty output
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(level)

    # Root logger configuration
    logging.basicConfig(
        level=level,
        handlers=[file_handler, console_handler]
    )

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
