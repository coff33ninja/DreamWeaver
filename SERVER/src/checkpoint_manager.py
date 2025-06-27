import os
import shutil
from datetime import datetime

import json
# Define base paths. In a larger application, these could come from a config file.
BASE_DATA_PATH = "E:/DreamWeaver/data"
BASE_CHECKPOINT_PATH = "E:/DreamWeaver/checkpoints"
DB_FILE_PATH = os.path.join(BASE_DATA_PATH, "dream_weaver.db")
# This path is specific to the server's LLM (PC1) and model name.
# A more advanced system might manage multiple server-side models.
SERVER_ADAPTER_PATH = os.path.join(BASE_DATA_PATH, "models/adapters/TinyLLaMA/PC1")

class CheckpointManager:
    def __init__(self):
        os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)

    def list_checkpoints(self):
        """Returns a list of available checkpoint names, sorted by most recent first."""
        try:
            return sorted([d for d in os.listdir(BASE_CHECKPOINT_PATH) if os.path.isdir(os.path.join(BASE_CHECKPOINT_PATH, d))], reverse=True)
        except FileNotFoundError:
            return []

    def save_checkpoint(self, name_prefix=""):
        """Saves the current state (DB and model adapters) to a new checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{name_prefix}_{timestamp}" if name_prefix else timestamp
        checkpoint_dir = os.path.join(BASE_CHECKPOINT_PATH, checkpoint_name)

        try:
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 1. Save the database
            if os.path.exists(DB_FILE_PATH):
                shutil.copy(DB_FILE_PATH, os.path.join(checkpoint_dir, "dream_weaver.db"))

            # 2. Save the server's LLM adapters
            if os.path.exists(SERVER_ADAPTER_PATH):
                shutil.copytree(SERVER_ADAPTER_PATH, os.path.join(checkpoint_dir, "PC1_adapters"))
            else:
                print(f"Warning: Server LLM adapters not found at {SERVER_ADAPTER_PATH}. Skipping adapter save.")

            return f"Checkpoint '{checkpoint_name}' saved successfully.", self.list_checkpoints()
        except Exception as e:
            return f"Error saving checkpoint: {e}", self.list_checkpoints()

    def load_checkpoint(self, checkpoint_name):
        """Restores the state from a given checkpoint."""
        checkpoint_dir = os.path.join(BASE_CHECKPOINT_PATH, checkpoint_name)
        if not os.path.isdir(checkpoint_dir):
            return f"Error: Checkpoint '{checkpoint_name}' not found."

        try:
            # 1. Restore the database
            shutil.copy(os.path.join(checkpoint_dir, "dream_weaver.db"), DB_FILE_PATH)

            # 2. Restore the server's LLM adapters
            if os.path.exists(os.path.join(checkpoint_dir, "PC1_adapters")): # Check if adapters exist in checkpoint
                if os.path.exists(SERVER_ADAPTER_PATH):
                    shutil.rmtree(SERVER_ADAPTER_PATH) # Remove existing adapters before copying
                shutil.copytree(os.path.join(checkpoint_dir, "PC1_adapters"), SERVER_ADAPTER_PATH)
            else:
                print(f"Warning: No PC1 adapters found in checkpoint '{checkpoint_name}'. Skipping adapter load.")

            return f"Checkpoint '{checkpoint_name}' loaded. PLEASE RESTART THE APPLICATION for changes to take effect."
        except Exception as e:
            return f"Error loading checkpoint: {e}"
    def export_story(self, export_format="text"):
        """Exports the story history to a file (text or JSON)."""
        from .database import Database # Local import to avoid circular dependency
        db = Database(DB_FILE_PATH)
        history = db.get_story_history()

        if not history:
            return "Error: No story history found."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(BASE_DATA_PATH, "exports")
        os.makedirs(export_dir, exist_ok=True)

        if export_format == "json":
            export_filename = f"story_export_{timestamp}.json"
            export_path = os.path.join(export_dir, export_filename)
            story_data = [{"speaker": entry[0], "text": entry[1], "timestamp": entry[2]} for entry in history]
            try:
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(story_data, f, indent=4, ensure_ascii=False)
                return f"Story exported to '{export_filename}' (JSON).", export_filename
            except Exception as e:
                return f"Error exporting story to JSON: {e}", None

        elif export_format == "text":
            export_filename = f"story_export_{timestamp}.txt"
            export_path = os.path.join(export_dir, export_filename)
            try:
                with open(export_path, "w", encoding="utf-8") as f:
                    for entry in history:
                        f.write(f"[{entry[2]}] {entry[0]}: {entry[1]}\n")
                return f"Story exported to '{export_filename}' (Text).", export_filename
            except Exception as e:
                return f"Error exporting story to Text: {e}", None

        else:
            return "Error: Invalid export format. Choose 'text' or 'json'.", None
