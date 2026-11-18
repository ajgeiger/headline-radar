"""Console logging used by the load process."""

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the project logger.

    Safe to call more than once: existing handlers are left in place.
    """
    logger = logging.getLogger("headline_radar")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger
