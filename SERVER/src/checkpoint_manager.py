import os
import shutil
from datetime import datetime
import json
from .config import DB_PATH, ADAPTERS_PATH, BASE_CHECKPOINT_PATH, BASE_DATA_PATH # Import from config

# Path for server-specific adapters (Actor1)
# Now configurable via CheckpointManager arguments for model and Actor ID.

class CheckpointManager:
    def __init__(self, server_model_name="TinyLLaMA", server_Actor_id="Actor1"):
        """
        Initialize a CheckpointManager for a specific server model and actor.
        
        Creates necessary directories for checkpoint storage and server-specific adapters based on the provided model and actor identifiers.
        
        Parameters:
            server_model_name (str): Name of the server model. Defaults to "TinyLLaMA".
            server_Actor_id (str): Identifier for the server actor. Defaults to "Actor1".
        """
        self.server_model_name = server_model_name
        self.server_Actor_id = server_Actor_id
        self.server_adapter_specific_path = os.path.join(ADAPTERS_PATH, self.server_model_name, self.server_Actor_id)
        os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)
        # Ensure the specific server adapter path for Actor1 also exists, as it's a target for copying
        os.makedirs(self.server_adapter_specific_path, exist_ok=True)


    def list_checkpoints(self):
        """
        Return a list of available checkpoint directory names, sorted from most recent to oldest.
        
        Returns:
            List of checkpoint names as strings, or an empty list if no checkpoint directory exists.
        """
        try:
            return sorted([d for d in os.listdir(BASE_CHECKPOINT_PATH) if os.path.isdir(os.path.join(BASE_CHECKPOINT_PATH, d))], reverse=True)
        except FileNotFoundError:
            return []

    def save_checkpoint(self, name_prefix=""):
        """
        Create a new checkpoint by saving the current database and server-specific model adapters.
        
        Parameters:
            name_prefix (str, optional): An optional prefix for the checkpoint directory name.
        
        Returns:
            tuple: A message indicating success or error, and the updated list of checkpoint names.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{name_prefix}_{timestamp}" if name_prefix else timestamp
        checkpoint_dir = os.path.join(BASE_CHECKPOINT_PATH, checkpoint_name)

        try:
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 1. Save the database
            if os.path.exists(DB_PATH): # Use DB_PATH from config
                shutil.copy(DB_PATH, os.path.join(checkpoint_dir, "dream_weaver.db"))

            # 2. Save the server's LLM adapters
            if os.path.exists(self.server_adapter_specific_path): # Use SERVER_ADAPTER_SPECIFIC_PATH
                shutil.copytree(self.server_adapter_specific_path, os.path.join(checkpoint_dir, f"{self.server_Actor_id}_adapters"))
            else:
                print(f"Warning: Server LLM adapters not found at {self.server_adapter_specific_path}. Skipping adapter save.")

            return f"Checkpoint '{checkpoint_name}' saved successfully.", self.list_checkpoints()
        except Exception as e:
            return f"Error saving checkpoint: {e}", self.list_checkpoints()

    def load_checkpoint(self, checkpoint_name):
        """
        Restore the database and server-specific adapters from a specified checkpoint.
        
        Parameters:
            checkpoint_name (str): The name of the checkpoint directory to restore from.
        
        Returns:
            str: A status message indicating success, error, or missing components. On success, instructs to restart the application for changes to take effect.
        """
        checkpoint_dir = os.path.join(BASE_CHECKPOINT_PATH, checkpoint_name)
        if not os.path.isdir(checkpoint_dir):
            return f"Error: Checkpoint '{checkpoint_name}' not found."

        try:
            # 1. Restore the database
            db_in_checkpoint = os.path.join(checkpoint_dir, "dream_weaver.db")
            if os.path.exists(db_in_checkpoint):
                shutil.copy(db_in_checkpoint, DB_PATH) # Use DB_PATH from config
            else:
                return f"Error: Database not found in checkpoint '{checkpoint_name}'."


            # 2. Restore the server's LLM adapters
            adapters_in_checkpoint = os.path.join(checkpoint_dir, f"{self.server_Actor_id}_adapters")
            if os.path.exists(adapters_in_checkpoint):
                if os.path.exists(self.server_adapter_specific_path): # Use SERVER_ADAPTER_SPECIFIC_PATH
                    shutil.rmtree(self.server_adapter_specific_path) # Remove existing adapters before copying
                shutil.copytree(adapters_in_checkpoint, self.server_adapter_specific_path) # Use SERVER_ADAPTER_SPECIFIC_PATH
            else:
                print(f"Warning: No {self.server_Actor_id} adapters found in checkpoint '{checkpoint_name}'. Skipping adapter load.")

            return f"Checkpoint '{checkpoint_name}' loaded. PLEASE RESTART THE APPLICATION for changes to take effect."
        except Exception as e:
            return f"Error loading checkpoint: {e}"

    def export_story(self, export_format="text"):
        """
        Export the story history to a file in either text or JSON format.
        
        Parameters:
            export_format (str): The format to export the story history in. Must be either "text" or "json".
        
        Returns:
            tuple: A tuple containing a status message and the export filename (or None if an error occurred).
        """
        from .database import Database # Local import to avoid circular dependency
        db = Database(DB_PATH) # Use DB_PATH from config
        history = db.get_story_history()

        if not history:
            return "Error: No story history found."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Exports will be saved in a subdirectory of BASE_DATA_PATH
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
