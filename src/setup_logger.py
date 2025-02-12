import logging
import os
from datetime import datetime


def setup_logger():
    """
    Set up a timestamped logger that logs to both a file and the console and return it.

    Parameters:
    -None

    Returns:
    -logger (logging.Logger): A configured logger instance
    """
    # check if the logger is already set up, if so return it, else set it up
    if len(logging.getLogger().handlers) > 0:
        return logging.getLogger(__name__)

    # Generate a timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join("./logs", f"run_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),  # Log to a new file for each run
            logging.StreamHandler(),  # Also log to console
        ],
    )

    # Create a global logger instance
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")

    return logger
