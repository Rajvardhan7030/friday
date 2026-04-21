"""Structured logging to local rotating files only."""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler


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
