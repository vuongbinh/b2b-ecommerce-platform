"""
Logging configuration for the application
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any

from .config import settings


def setup_logging() -> None:
    """Setup application logging configuration"""
    
    # Create logs directory if it doesn't exist
    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Logging configuration
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "standard",
                "stream": sys.stdout
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": settings.LOG_LEVEL,
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            },
            "uvicorn.error": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            },
            "sqlalchemy.engine": {
                "handlers": ["console"],
                "level": "WARNING" if not settings.is_development else "INFO",
                "propagate": False
            }
        }
    }
    
    # Add file handler if log file is specified
    if settings.LOG_FILE:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "detailed",
            "filename": settings.LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        
        # Add file handler to all loggers
        for logger_config in config["loggers"].values():
            if isinstance(logger_config["handlers"], list):
                logger_config["handlers"].append("file")
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Set third-party loggers to WARNING level in production
    if settings.is_production:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)