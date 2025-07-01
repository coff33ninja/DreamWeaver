import os
from .config import PROJECT_ROOT

# Path to the .env file in the project root
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")

def get_env_file_status():
    """
    Check whether the `.env` file exists at the project root and return a status message.
    
    Returns:
        str: Message indicating if the `.env` file is present or will be created upon saving.
    """
    if os.path.exists(ENV_FILE_PATH):
        return f".env file found at {ENV_FILE_PATH}"
    return f".env file not found. It will be created at {ENV_FILE_PATH} upon saving."

def load_env_vars(mask_sensitive=False):
    """
    Load environment variables from the `.env` file as a dictionary.
    
    If `mask_sensitive` is True, values for keys containing "TOKEN", "KEY", or "SECRET" (case-insensitive) are masked for security. Returns an empty dictionary if the `.env` file does not exist.
    
    Parameters:
        mask_sensitive (bool): Whether to mask sensitive values in the returned dictionary.
    
    Returns:
        dict: Dictionary of environment variable key-value pairs, with sensitive values masked if requested.
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
    Merge new environment variable assignments with existing ones and save the result to the `.env` file.
    
    Parameters:
        new_vars_str (str): String containing environment variable assignments, one per line in `KEY=VALUE` format.
    
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

        return "Successfully saved to .env file. A server restart is required for changes to take effect."
    except Exception as e:
        return f"Error saving to .env file: {e}"