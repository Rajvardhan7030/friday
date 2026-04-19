"""Structured logging to local rotating files only."""

import logging
import logging.handlers
from pathlib import Path
from rich.logging import RichHandler

def setup_logging(log_file: Path, level: int = logging.INFO) -> None:
    """Setup structured logging to a local rotating file and console."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Use a structured format for the log file
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Rotating file handler (10MB per file, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

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
