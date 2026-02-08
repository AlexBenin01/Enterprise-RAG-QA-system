"""Centralized logging configuration."""
import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from pythonjsonlogger.json import JsonFormatter as jsonlogger

from .config import settings


def setup_logging() -> None:
    """Configure structured logging with JSON output."""
    
    # Create logs directory
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure standard logging
    log_level = logging.DEBUG if settings.debug else logging.INFO
    
    # JSON formatter for file output
    json_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    
    # File handler with JSON
    file_handler = logging.FileHandler(settings.logs_dir / "app.log")
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(log_level)
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Get a logger instance."""
    return structlog.get_logger(name)
