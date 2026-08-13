"""
Sanitized Error Handling & Logging System for Ardhanarishvara.
Enforces MANDATORY SECURITY REQUIREMENT #5:
- Internal detailed stack traces are logged privately to log files.
- User-facing outputs receive sanitized, safe error messages without internal system path disclosure.
"""

import logging
import os
import sys
import traceback
from functools import wraps

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "system_internal.log")

# Configure private logger
logger = logging.getLogger("ArdhanarishvaraInternalLogger")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class ArdhanarishvaraSecurityException(Exception):
    """Base sanitized security exception."""
    pass


class InvalidDataFormatException(ArdhanarishvaraSecurityException):
    """Raised when data format or schema validation fails."""
    pass


class FileUploadSecurityException(ArdhanarishvaraSecurityException):
    """Raised when file validation fails."""
    pass


class RateLimitExceededException(ArdhanarishvaraSecurityException):
    """Raised when API rate limit is reached."""
    pass


def sanitize_errors(user_message="An internal error occurred during data processing."):
    """Decorator to catch exceptions, log raw tracebacks privately, and re-raise a sanitized message."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ArdhanarishvaraSecurityException as se:
                # Security exceptions are already sanitized for users
                logger.warning(f"Sanitized security event in {func.__name__}: {str(se)}")
                raise
            except Exception as e:
                # Log full un-truncated traceback to private internal log file
                tb_str = traceback.format_exc()
                logger.error(f"Internal exception in {func.__name__}: {str(e)}\n{tb_str}")
                # Raise sanitized error to end user
                raise RuntimeError(f"[Ardhanarishvara Safety]: {user_message}") from None
        return wrapper
    return decorator


def log_info(msg: str):
    """Log non-sensitive informative operational status."""
    logger.info(msg)
    print(f"[INFO] {msg}")
