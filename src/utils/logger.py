from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


def get_logger(module_name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Features
    - Console + file logging
    - Log rotation
    - Safe path handling
    - Prevents duplicate handlers
    """

    logger = logging.getLogger(module_name)

    # Prevent reconfiguration if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # ------------------------------------------------------------------
    # Logs directory
    # ------------------------------------------------------------------

    base_dir = Path(__file__).resolve().parents[2]
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{datetime.now():%Y-%m-%d}_system.log"

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    console_handler = logging.StreamHandler()

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8"
    )

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Formatter
    # ------------------------------------------------------------------

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
