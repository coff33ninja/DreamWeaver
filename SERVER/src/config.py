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
MODELS_PATH = os.getenv("DREAMWEAVER_MODEL_PATH", os.path.join(BASE_DATA_PATH, "models"))
ADAPTERS_PATH = os.path.join(MODELS_PATH, "adapters")

# Checkpoints path
BASE_CHECKPOINT_PATH = os.getenv("DREAMWEAVER_CHECKPOINT_PATH", os.path.join(PROJECT_ROOT, "checkpoints"))

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
