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


# --- Function to Create Directories ---
def ensure_client_directories():
    """
    Ensure that all required directories for the CharacterClient exist, creating them if necessary.
    
    If a directory cannot be created due to an OS error (such as insufficient permissions), a warning is printed but execution continues.
    """
    paths_to_create = [
        CLIENT_DATA_PATH,
        CLIENT_MODELS_PATH,
        CLIENT_LLM_MODELS_PATH,
        CLIENT_TTS_MODELS_PATH,
        CLIENT_TTS_REFERENCE_VOICES_PATH,
        CLIENT_LOGS_PATH,
        CLIENT_TEMP_AUDIO_PATH
    ]
    for path in paths_to_create:
        try:
            os.makedirs(path, exist_ok=True)
            # print(f"Ensured directory exists: {path}") # Optional: for debugging
        except OSError as e:
            print(f"Error creating directory {path}: {e}. Please check permissions.")
            # Depending on severity, you might want to raise an error here or handle it.
            # For now, just printing an error.

# --- Run directory creation when this module is loaded ---
ensure_client_directories()


if __name__ == "__main__":
    # Helper to print out the defined paths for verification if script is run directly
    print(f"CLIENT_ROOT: {CLIENT_ROOT}")
    print(f"CLIENT_DATA_PATH: {CLIENT_DATA_PATH} (Default was: {DEFAULT_CLIENT_DATA_PATH})")
    print(f"CLIENT_MODELS_PATH: {CLIENT_MODELS_PATH} (Default was: {DEFAULT_CLIENT_MODELS_PATH})")
    print(f"  CLIENT_LLM_MODELS_PATH: {CLIENT_LLM_MODELS_PATH}")
    print(f"  CLIENT_TTS_MODELS_PATH: {CLIENT_TTS_MODELS_PATH}")
    print(f"  CLIENT_TTS_REFERENCE_VOICES_PATH: {CLIENT_TTS_REFERENCE_VOICES_PATH}")
    print(f"CLIENT_LOGS_PATH: {CLIENT_LOGS_PATH}")
    print(f"CLIENT_TEMP_AUDIO_PATH: {CLIENT_TEMP_AUDIO_PATH}")
    print("\nNote: If you see default paths, environment variables for overrides were not set or found.")
