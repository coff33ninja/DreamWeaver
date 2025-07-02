import os

# Determine the CharacterClient root directory, assuming this config.py is in CharacterClient/src
CLIENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Base Data Path ---
# Users can override this by setting the DREAMWEAVER_CLIENT_DATA_PATH environment variable.
# Default is 'CharacterClient/data/' relative to the client root.
DEFAULT_CLIENT_DATA_PATH = os.path.join(CLIENT_ROOT, "data")
CLIENT_DATA_PATH = os.getenv("DREAMWEAVER_CLIENT_DATA_PATH", DEFAULT_CLIENT_DATA_PATH)

# --- Models Path ---
# Users can override this by setting the DREAMWEAVER_CLIENT_MODELS_PATH environment variable.
# Default is '[CLIENT_DATA_PATH]/models/'.
DEFAULT_CLIENT_MODELS_PATH = os.path.join(CLIENT_DATA_PATH, "models")
CLIENT_MODELS_PATH = os.getenv("DREAMWEAVER_CLIENT_MODELS_PATH", DEFAULT_CLIENT_MODELS_PATH)

# Specific model type paths within CLIENT_MODELS_PATH
CLIENT_LLM_MODELS_PATH = os.path.join(CLIENT_MODELS_PATH, "llm")
CLIENT_TTS_MODELS_PATH = os.path.join(CLIENT_MODELS_PATH, "tts")
CLIENT_TTS_REFERENCE_VOICES_PATH = os.path.join(CLIENT_TTS_MODELS_PATH, "reference_voices")

# --- Logs Path ---
# Default is '[CLIENT_DATA_PATH]/logs/'.
CLIENT_LOGS_PATH = os.path.join(CLIENT_DATA_PATH, "logs")

# --- Temporary Audio Path ---
# For storing synthesized audio before sending to server, or other temp files.
# Default is '[CLIENT_DATA_PATH]/temp_audio/'.
CLIENT_TEMP_AUDIO_PATH = os.path.join(CLIENT_DATA_PATH, "temp_audio")


import logging

# --- Function to Create Directories ---
def ensure_client_directories():
    """
    Ensure that all required directories for the client application exist, creating them if necessary.
    Uses a basic logger for this pre-initialization phase, as the main client logger might not be set up yet.
    """
    # Use a temporary logger for this specific function, as it runs at import time.
    # This avoids dependency on the full logging_config being initialized if this module is imported first.
    temp_logger = logging.getLogger("dreamweaver_client_config_setup")
    if not temp_logger.hasHandlers(): # Configure only if not already configured (e.g. by another import)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        temp_logger.addHandler(handler)
        temp_logger.setLevel(logging.INFO) # Or WARNING for less verbosity

    paths_to_create = [
        CLIENT_DATA_PATH,
        CLIENT_MODELS_PATH,
        CLIENT_LLM_MODELS_PATH,
        CLIENT_TTS_MODELS_PATH,
        CLIENT_TTS_REFERENCE_VOICES_PATH,
        CLIENT_LOGS_PATH, # This directory is for the main logger's file handler
        CLIENT_TEMP_AUDIO_PATH
    ]
    temp_logger.info("Ensuring client directories exist...")
    for path in paths_to_create:
        try:
            os.makedirs(path, exist_ok=True)
            temp_logger.debug(f"Ensured directory exists: {path}")
        except OSError as e:
            # Log to stderr as well, as this is a critical setup step.
            error_message = f"CRITICAL ERROR creating directory {path}: {e}. Please check permissions. Client may not function correctly."
            temp_logger.critical(error_message)
            import sys
            print(error_message, file=sys.stderr) # Also print to stderr for immediate visibility

# --- Run directory creation when this module is loaded ---
ensure_client_directories()


if __name__ == "__main__":
    # Setup a basic logger for standalone execution of this config file for verification
    main_logger_name = "dreamweaver_client_config_test"
    config_test_logger = logging.getLogger(main_logger_name)
    if not config_test_logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        config_test_logger.addHandler(ch)
        config_test_logger.setLevel(logging.INFO)

    config_test_logger.info(f"CLIENT_ROOT: {CLIENT_ROOT}")
    config_test_logger.info(f"CLIENT_DATA_PATH: {CLIENT_DATA_PATH} (Default was: {DEFAULT_CLIENT_DATA_PATH})")
    config_test_logger.info(f"CLIENT_MODELS_PATH: {CLIENT_MODELS_PATH} (Default was: {DEFAULT_CLIENT_MODELS_PATH})")
    config_test_logger.info(f"  CLIENT_LLM_MODELS_PATH: {CLIENT_LLM_MODELS_PATH}")
    config_test_logger.info(f"  CLIENT_TTS_MODELS_PATH: {CLIENT_TTS_MODELS_PATH}")
    config_test_logger.info(f"  CLIENT_TTS_REFERENCE_VOICES_PATH: {CLIENT_TTS_REFERENCE_VOICES_PATH}")
    config_test_logger.info(f"CLIENT_LOGS_PATH: {CLIENT_LOGS_PATH}")
    config_test_logger.info(f"CLIENT_TEMP_AUDIO_PATH: {CLIENT_TEMP_AUDIO_PATH}")
    config_test_logger.info("\nNote: If you see default paths, environment variables for overrides were not set or found.")
