import os

# Determine the project root directory, assuming this config.py is in SERVER/src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Base path for all data
BASE_DATA_PATH = os.getenv("DREAMWEAVER_DATA_PATH", os.path.join(PROJECT_ROOT, "data"))

# Database path
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DATA_PATH, "dream_weaver.db"))

# Audio paths
AUDIO_PATH = os.path.join(BASE_DATA_PATH, "audio")
NARRATOR_AUDIO_PATH = os.path.join(AUDIO_PATH, "narrator")
CHARACTERS_AUDIO_PATH = os.path.join(AUDIO_PATH, "characters")
REFERENCE_VOICES_AUDIO_PATH = os.path.join(AUDIO_PATH, "reference_voices")

# Models paths
MODELS_PATH = os.getenv(
    "DREAMWEAVER_MODEL_PATH", os.path.join(BASE_DATA_PATH, "models")
)
ADAPTERS_PATH = os.path.join(MODELS_PATH, "adapters")

# Checkpoints path
BASE_CHECKPOINT_PATH = os.getenv(
    "DREAMWEAVER_CHECKPOINT_PATH", os.path.join(PROJECT_ROOT, "checkpoints")
)

# --- DreamWeaver Configurable Options ---
DEFAULT_WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
DIARIZATION_ENABLED = os.getenv("DIARIZATION_ENABLED", "1") == "1"

# Additional narrator/model config options
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
AUDIO_FORMAT = os.getenv("AUDIO_FORMAT", ".wav")
MAX_DIARIZATION_RETRIES = int(os.getenv("MAX_DIARIZATION_RETRIES", "3"))

# Server's Actor1 specific configuration
ACTOR1_PYGAME_AUDIO_ENABLED = (
    os.getenv("ACTOR1_PYGAME_AUDIO_ENABLED", "0") == "1"
)  # Default to False (disabled)

# Session duration for issued tokens (in hours)
SESSION_DURATION_HOURS = int(os.getenv("SESSION_DURATION_HOURS", 1))

# List of editable config options for UI
EDITABLE_CONFIG_OPTIONS = {
    "WHISPER_MODEL_SIZE": DEFAULT_WHISPER_MODEL_SIZE,
    "DIARIZATION_ENABLED": DIARIZATION_ENABLED,
    "DIARIZATION_MODEL": DIARIZATION_MODEL,
    "AUDIO_FORMAT": AUDIO_FORMAT,
    "MAX_DIARIZATION_RETRIES": MAX_DIARIZATION_RETRIES,
    "ACTOR1_PYGAME_AUDIO_ENABLED": ACTOR1_PYGAME_AUDIO_ENABLED,
    "SESSION_DURATION_HOURS": SESSION_DURATION_HOURS,
    # Add more as needed
}

# Ensure necessary directories exist
os.makedirs(BASE_DATA_PATH, exist_ok=True)
os.makedirs(AUDIO_PATH, exist_ok=True)
os.makedirs(NARRATOR_AUDIO_PATH, exist_ok=True)
os.makedirs(CHARACTERS_AUDIO_PATH, exist_ok=True)
os.makedirs(REFERENCE_VOICES_AUDIO_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(ADAPTERS_PATH, exist_ok=True)
os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)

if __name__ == "__main__":
    # Helper to print out the defined paths for verification
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"BASE_DATA_PATH: {BASE_DATA_PATH}")
    print(f"DB_PATH: {DB_PATH}")
    print(f"AUDIO_PATH: {AUDIO_PATH}")
    print(f"NARRATOR_AUDIO_PATH: {NARRATOR_AUDIO_PATH}")
    print(f"CHARACTERS_AUDIO_PATH: {CHARACTERS_AUDIO_PATH}")
    print(f"REFERENCE_VOICES_AUDIO_PATH: {REFERENCE_VOICES_AUDIO_PATH}")
    print(f"MODELS_PATH: {MODELS_PATH}")
    print(f"ADAPTERS_PATH: {ADAPTERS_PATH}")
    print(f"BASE_CHECKPOINT_PATH: {BASE_CHECKPOINT_PATH}")
    print(f"DEFAULT_WHISPER_MODEL_SIZE: {DEFAULT_WHISPER_MODEL_SIZE}")
    print(f"DIARIZATION_ENABLED: {DIARIZATION_ENABLED}")
    print(f"ACTOR1_PYGAME_AUDIO_ENABLED: {ACTOR1_PYGAME_AUDIO_ENABLED}")
    print(f"SESSION_DURATION_HOURS: {SESSION_DURATION_HOURS}")
    print(f"EDITABLE_CONFIG_OPTIONS: {EDITABLE_CONFIG_OPTIONS}")
