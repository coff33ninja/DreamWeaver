import os
import shutil
from datetime import datetime
import json
from .config import (
    DB_PATH,
    ADAPTERS_PATH,
    BASE_CHECKPOINT_PATH,
    BASE_DATA_PATH,
)  # Import from config
import logging

logger = logging.getLogger("dreamweaver_server")

# Path for server-specific adapters (Actor1)
# Now configurable via CheckpointManager arguments for model and Actor ID.


class CheckpointManager:
    def __init__(self, server_model_name="TinyLLaMA", server_Actor_id="Actor1"):
        """
        Initialize the CheckpointManager with configurable server model and actor identifiers.

        Creates necessary directories for storing checkpoints and server-specific adapters based on the provided model and actor IDs.
        """
        self.server_model_name = server_model_name
        self.server_Actor_id = server_Actor_id
        self.server_adapter_specific_path = os.path.join(
            ADAPTERS_PATH, self.server_model_name, self.server_Actor_id
        )

        try:
            os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)
            os.makedirs(self.server_adapter_specific_path, exist_ok=True)
            logger.info(
                f"CheckpointManager initialized. Checkpoint path: {BASE_CHECKPOINT_PATH}, Server adapter path: {self.server_adapter_specific_path}"
            )
        except OSError as e:
            logger.error(
                f"Error creating directories during CheckpointManager initialization: {e}",
                exc_info=True,
            )
            # Depending on severity, might re-raise or handle differently

    def list_checkpoints(self):
        """
        Return a list of available checkpoint directory names, sorted from most recent to oldest.

        Returns:
            List of checkpoint names as strings. Returns an empty list if the checkpoint directory does not exist.
        """
        try:
            checkpoints = sorted(
                [
                    d
                    for d in os.listdir(BASE_CHECKPOINT_PATH)
                    if os.path.isdir(os.path.join(BASE_CHECKPOINT_PATH, d))
                ],
                key=lambda x: os.path.getmtime(
                    os.path.join(BASE_CHECKPOINT_PATH, x)
                ),  # Sort by modification time
                reverse=True,
            )
            logger.debug(f"Found checkpoints: {checkpoints}")
            return checkpoints
        except FileNotFoundError:
            logger.warning(
                f"Checkpoint directory {BASE_CHECKPOINT_PATH} not found when listing checkpoints."
            )
            return []
        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}", exc_info=True)
            return []

    def save_checkpoint(self, name_prefix=""):
        """
        Create a new checkpoint by saving the current database and server adapter files to a timestamped directory.

        Parameters:
            name_prefix (str, optional): An optional prefix for the checkpoint directory name.

        Returns:
            tuple: A message indicating success or failure, and the updated list of checkpoint names.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{name_prefix}_{timestamp}" if name_prefix else timestamp
        checkpoint_dir = os.path.join(BASE_CHECKPOINT_PATH, checkpoint_name)
        logger.info(
            f"Attempting to save checkpoint: {checkpoint_name} to {checkpoint_dir}"
        )

        try:
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 1. Save the database
            if os.path.exists(DB_PATH):  # Use DB_PATH from config
                shutil.copy(DB_PATH, os.path.join(checkpoint_dir, "dream_weaver.db"))
                logger.info(f"Database saved to checkpoint {checkpoint_name}")
            else:
                logger.warning(
                    f"Database file not found at {DB_PATH}. Skipping DB save for checkpoint {checkpoint_name}."
                )

            # 2. Save the server's LLM adapters
            if os.path.exists(
                self.server_adapter_specific_path
            ):  # Use SERVER_ADAPTER_SPECIFIC_PATH
                shutil.copytree(
                    self.server_adapter_specific_path,
                    os.path.join(checkpoint_dir, f"{self.server_Actor_id}_adapters"),
                )
                logger.info(
                    f"Server LLM adapters for {self.server_Actor_id} saved to checkpoint {checkpoint_name}"
                )
            else:
                logger.warning(
                    f"Server LLM adapters not found at {self.server_adapter_specific_path}. Skipping adapter save for checkpoint {checkpoint_name}."
                )

            return (
                f"Checkpoint '{checkpoint_name}' saved successfully.",
                self.list_checkpoints(),
            )
        except Exception as e:
            logger.error(
                f"Error saving checkpoint {checkpoint_name}: {e}", exc_info=True
            )
            # Attempt to clean up partially created checkpoint directory
            if os.path.isdir(checkpoint_dir):
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(
                        f"Cleaned up partially created checkpoint directory: {checkpoint_dir}"
                    )
                except Exception as cleanup_e:
                    logger.error(
                        f"Error cleaning up checkpoint directory {checkpoint_dir}: {cleanup_e}",
                        exc_info=True,
                    )
            return f"Error saving checkpoint: {e}", self.list_checkpoints()

    def load_checkpoint(self, checkpoint_name):
        """
        Restore the database and server adapter state from a specified checkpoint.

        Parameters:
            checkpoint_name (str): The name of the checkpoint directory to restore from.

        Returns:
            str: A status message indicating success, the need to restart the application, or an error description if restoration fails or required files are missing.
        """
        checkpoint_dir = os.path.join(BASE_CHECKPOINT_PATH, checkpoint_name)
        logger.info(
            f"Attempting to load checkpoint: {checkpoint_name} from {checkpoint_dir}"
        )
        if not os.path.isdir(checkpoint_dir):
            logger.error(f"Checkpoint directory '{checkpoint_dir}' not found.")
            return f"Error: Checkpoint '{checkpoint_name}' not found."

        try:
            # 1. Restore the database
            db_in_checkpoint = os.path.join(checkpoint_dir, "dream_weaver.db")
            if os.path.exists(db_in_checkpoint):
                shutil.copy(db_in_checkpoint, DB_PATH)  # Use DB_PATH from config
                logger.info(f"Database restored from checkpoint {checkpoint_name}")
            else:
                logger.error(
                    f"Database file not found in checkpoint '{checkpoint_name}'."
                )
                return f"Error: Database not found in checkpoint '{checkpoint_name}'."

            # 2. Restore the server's LLM adapters
            adapters_in_checkpoint = os.path.join(
                checkpoint_dir, f"{self.server_Actor_id}_adapters"
            )
            if os.path.exists(adapters_in_checkpoint):
                if os.path.exists(
                    self.server_adapter_specific_path
                ):  # Use SERVER_ADAPTER_SPECIFIC_PATH
                    logger.info(
                        f"Removing existing server adapters at {self.server_adapter_specific_path} before restoring from checkpoint."
                    )
                    shutil.rmtree(self.server_adapter_specific_path)
                shutil.copytree(
                    adapters_in_checkpoint, self.server_adapter_specific_path
                )  # Use SERVER_ADAPTER_SPECIFIC_PATH
                logger.info(
                    f"Server LLM adapters for {self.server_Actor_id} restored from checkpoint {checkpoint_name}"
                )
            else:
                logger.warning(
                    f"No {self.server_Actor_id} adapters found in checkpoint '{checkpoint_name}'. Skipping adapter load."
                )

            return f"Checkpoint '{checkpoint_name}' loaded. PLEASE RESTART THE APPLICATION for changes to take effect."
        except Exception as e:
            logger.error(
                f"Error loading checkpoint {checkpoint_name}: {e}", exc_info=True
            )
            return f"Error loading checkpoint: {e}"

    def export_story(self, export_format="text"):
        """
        Export the story history to a file in either text or JSON format.

        Parameters:
            export_format (str): The format to export the story history in. Must be either "text" or "json".

        Returns:
            tuple: A tuple containing a status message and the filename if successful, or an error message and None if unsuccessful.
        """
        logger.info(f"Attempting to export story in {export_format} format.")
        from .database import (
            Database,
        )  # Local import to avoid circular dependency issues with logging setup

        db = Database(DB_PATH)  # Use DB_PATH from config
        history = (
            db.get_story_history()
        )  # This method in DB should ideally use its own logger

        if not history:
            logger.warning("No story history found to export.")
            return "Error: No story history found.", None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Exports will be saved in a subdirectory of BASE_DATA_PATH
        export_dir = os.path.join(BASE_DATA_PATH, "exports")

        try:
            os.makedirs(export_dir, exist_ok=True)
        except OSError as e:
            logger.error(
                f"Error creating export directory {export_dir}: {e}", exc_info=True
            )
            return f"Error creating export directory: {e}", None

        if export_format == "json":
            export_filename = f"story_export_{timestamp}.json"
            export_path = os.path.join(export_dir, export_filename)
            # Assuming history items are dicts or sqlite3.Row which behave like dicts
            story_data = [
                {
                    "speaker": entry["speaker"],
                    "text": entry["text"],
                    "timestamp": entry["timestamp"],
                }
                for entry in history
            ]
            try:
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(story_data, f, indent=4, ensure_ascii=False)
                logger.info(f"Story exported to '{export_path}' (JSON).")
                return f"Story exported to '{export_filename}' (JSON).", export_filename
            except Exception as e:
                logger.error(
                    f"Error exporting story to JSON file {export_path}: {e}",
                    exc_info=True,
                )
                return f"Error exporting story to JSON: {e}", None

        elif export_format == "text":
            export_filename = f"story_export_{timestamp}.txt"
            export_path = os.path.join(export_dir, export_filename)
            try:
                with open(export_path, "w", encoding="utf-8") as f:
                    for (
                        entry
                    ) in history:  # Assuming history items are dicts or sqlite3.Row
                        f.write(
                            f"[{entry['timestamp']}] {entry['speaker']}: {entry['text']}\n"
                        )
                logger.info(f"Story exported to '{export_path}' (Text).")
                return f"Story exported to '{export_filename}' (Text).", export_filename
            except Exception as e:
                logger.error(
                    f"Error exporting story to Text file {export_path}: {e}",
                    exc_info=True,
                )
                return f"Error exporting story to Text: {e}", None

        else:
            logger.warning(f"Invalid export format requested: {export_format}")
            return "Error: Invalid export format. Choose 'text' or 'json'.", None
