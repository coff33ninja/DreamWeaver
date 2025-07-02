import logging
import logging.handlers
import os

# Define the root path for logs within the SERVER directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs") # SERVER/logs/
SERVER_LOG_FILE = os.path.join(LOGS_DIR, "server.log")

DEFAULT_LOG_LEVEL = logging.INFO
LOGGER_NAME = "dreamweaver_server"

def setup_logging(log_level=DEFAULT_LOG_LEVEL):
    """
    Configures logging for the DreamWeaver server application.

    Sets up a logger that outputs to both the console and a rotating file.
    The log file is stored in 'SERVER/logs/server.log'.
    """
    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)

    # Prevent multiple handlers if setup_logging is called more than once (e.g., in tests or reloads)
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
        file_handler = logging.handlers.RotatingFileHandler(
            SERVER_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8' # 5MB per file, 3 backups
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # If file handler fails (e.g. permissions), log to console and continue
        logger.error(f"Failed to set up file logging for {SERVER_LOG_FILE}: {e}", exc_info=True)


    logger.info(f"Logging configured for {LOGGER_NAME}. Level: {logging.getLevelName(log_level)}. Output to console and {SERVER_LOG_FILE}")

def get_logger(name=LOGGER_NAME):
    """
    Returns the configured application logger.
    """
    return logging.getLogger(name)

if __name__ == "__main__":
    # Example usage:
    setup_logging(logging.DEBUG)
    logger = get_logger()
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    print(f"Log file should be at: {SERVER_LOG_FILE}")
