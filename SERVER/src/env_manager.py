import os
from typing import Dict
from .config import PROJECT_ROOT
import logging

logger = logging.getLogger("dreamweaver_server")

# Path to the .env file in the project root
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env") # This should be SERVER_PROJECT_ROOT if .env is in SERVER/
# Let's assume .env is meant to be in the overall project root (DreamWeaver/.env)
# If it's SERVER/.env, then PROJECT_ROOT in config.py (which is SERVER dir) is correct.
# For now, assuming PROJECT_ROOT from config.py is the intended location for .env (i.e. SERVER/.env)

def get_env_file_status():
    """
    Check whether the `.env` file exists at the project root and return a status message.
    
    Returns:
        str: A message indicating if the `.env` file is present or will be created upon saving.
    """
    if os.path.exists(ENV_FILE_PATH):
        msg = f".env file found at {ENV_FILE_PATH}"
        logger.info(msg)
        return msg
    msg = f".env file not found. It will be created at {ENV_FILE_PATH} upon saving."
    logger.info(msg)
    return msg

def load_env_vars(*, mask_sensitive: bool = False) -> Dict[str, str]:
    """
    Load environment variables from the `.env` file as a dictionary.
    
    If `mask_sensitive` is True, values for keys containing "TOKEN", "KEY", or "SECRET" (case-insensitive) are masked for security. Returns an empty dictionary if the `.env` file does not exist.
    
    Parameters:
        mask_sensitive (bool): Whether to mask sensitive values in the output.
    
    Returns:
        dict: Dictionary of environment variable key-value pairs.
    """
    env_vars = {}
    if not os.path.exists(ENV_FILE_PATH):
        logger.debug(f".env file not found at {ENV_FILE_PATH} during load_env_vars.")
        return env_vars

    logger.debug(f"Loading .env file from {ENV_FILE_PATH}")
    try:
        with open(ENV_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if mask_sensitive and ('TOKEN' in key.upper() or 'KEY' in key.upper() or 'SECRET' in key.upper()):
                    value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"

                env_vars[key] = value
    except Exception as e:
        logger.error(f"Error reading .env file at {ENV_FILE_PATH}: {e}", exc_info=True)
        # Return whatever was loaded so far, or an empty dict if critical
    return env_vars

def save_env_vars(new_vars_str: str):
    """
    Update or add environment variables in the `.env` file using a string of key-value assignments.
    
    Parameters:
        new_vars_str (str): String containing environment variable assignments, each in `KEY=VALUE` format on a new line.
    
    Returns:
        str: Success message if variables are saved, or an error message if saving fails.
    """
    try:
        existing_vars = load_env_vars(mask_sensitive=False)

        for line in new_vars_str.strip().split('\n'):
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key:
                    existing_vars[key] = value

        with open(ENV_FILE_PATH, "w", encoding="utf-8") as f:
            for key, value in existing_vars.items():
                f.write(f"{key}={value}\n")

        msg = "Successfully saved to .env file. A server restart is required for changes to take effect."
        logger.info(f"Saved variables to {ENV_FILE_PATH}. New content (masked for sensitive keys): {load_env_vars(mask_sensitive=True)}")
        return msg
    except Exception as e:
        logger.error(f"Error saving to .env file at {ENV_FILE_PATH}: {e}", exc_info=True)
        return f"Error saving to .env file: {e}"