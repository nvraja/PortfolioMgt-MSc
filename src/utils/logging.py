# src/utils/logging.py
"""
Logging convenience wrapper.
Usage:
    from src.utils.logging import configure_logging, get_logger
    configure_logging(level="INFO")
    log = get_logger(__name__)
"""

import logging
import sys
from typing import Optional

def configure_logging(level: str = "INFO", fmt: Optional[str] = None):
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, level.upper(), logging.INFO), format=fmt)

def get_logger(name: str):
    return logging.getLogger(name)
