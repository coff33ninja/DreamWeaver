import logging
import logging.handlers
import os
from .config import CLIENT_LOGS_PATH # Use the path from client's config

# Define the log file path using the imported CLIENT_LOGS_PATH
CLIENT_LOG_FILE = os.path.join(CLIENT_LOGS_PATH, "client.log")

DEFAULT_LOG_LEVEL = logging.INFO
LOGGER_NAME = "dreamweaver_client"

def setup_client_logging(log_level=DEFAULT_LOG_LEVEL):
    """
    Configures logging for the DreamWeaver Character Client application.

    Sets up a logger that outputs to both the console and a rotating file.
    The log file is stored in the directory specified by CLIENT_LOGS_PATH (e.g., CharacterClient/data/logs/client.log).
    """
    # config.ensure_client_directories() should have already created CLIENT_LOGS_PATH
    # but a check here doesn't hurt, though it might be redundant if called after config import.
    # For simplicity, we assume CLIENT_LOGS_PATH exists as per config.py's import-time execution.

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)

    # Prevent multiple handlers if setup_logging is called more than once
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler
    try:
        # Ensure the directory for the log file exists, just in case.
        # CLIENT_LOGS_PATH is the directory, CLIENT_LOG_FILE is the file path.
        os.makedirs(os.path.dirname(CLIENT_LOG_FILE), exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            CLIENT_LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding='utf-8' # 2MB per file, 3 backups
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"Failed to set up file logging for {CLIENT_LOG_FILE}: {e}", exc_info=True)

    logger.info(f"Client logging configured for {LOGGER_NAME}. Level: {logging.getLevelName(log_level)}. Output to console and {CLIENT_LOG_FILE}")

def get_logger(name=LOGGER_NAME):
    """
    Returns the configured application logger for the client.
    """
    return logging.getLogger(name)

if __name__ == "__main__":
    # This part is for testing the logging_config.py itself.
    # It requires config.py to be available in the path.
    # To run this directly for testing, you might need to adjust Python's path or run as a module.
    print(f"Attempting to set up client logging for testing (log file: {CLIENT_LOG_FILE})...")
    setup_client_logging(logging.DEBUG)
    logger_test = get_logger()
    logger_test.debug("This is a client debug message.")
    logger_test.info("This is a client info message.")
    logger_test.warning("This is a client warning message.")
    logger_test.error("This is a client error message.")
    logger_test.critical("This is a client critical message.")
    print(f"Test complete. Check console and log file: {CLIENT_LOG_FILE}")
