import os
from typing import Dict
from .config import PROJECT_ROOT

# Path to the .env file in the project root
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")

def get_env_file_status():
    """Checks if the .env file exists and returns its status."""
    if os.path.exists(ENV_FILE_PATH):
        return f".env file found at {ENV_FILE_PATH}"
    return f".env file not found. It will be created at {ENV_FILE_PATH} upon saving."

def load_env_vars(*, mask_sensitive: bool = False) -> Dict[str, str]:
    """
    Loads environment variables from the .env file.

    Args:
        mask_sensitive (bool): If True, masks values of keys containing 'TOKEN', 'KEY', or 'SECRET'.

    Returns:
        dict: A dictionary of environment variables.
    """
    env_vars = {}
    if not os.path.exists(ENV_FILE_PATH):
        return env_vars

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
    return env_vars

def save_env_vars(new_vars_str: str):
    """
    Saves or updates environment variables in the .env file from a string.
    Each new variable should be on a new line, e.g., "KEY1=VALUE1\nKEY2=VALUE2".
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

        return "Successfully saved to .env file. A server restart is required for changes to take effect."
    except Exception as e:
        return f"Error saving to .env file: {e}"