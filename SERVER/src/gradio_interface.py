import gradio as gr
from .csm import CSM
from .database import Database
from .tts_manager import TTSManager # Server's TTSManager
from .client_manager import ClientManager
from .checkpoint_manager import CheckpointManager
from . import env_manager
from .config import DB_PATH, REFERENCE_VOICES_AUDIO_PATH
import shutil
import os
import asyncio # Added asyncio
import sys
import logging

logger = logging.getLogger("dreamweaver_server")

# --- Instances ---
# These are now initialized inside launch_interface for process safety.
db_instance = None
client_manager_instance = None
csm_instance = None
checkpoint_manager = None
env_manager_instance = None # Not really an instance, but follows the pattern

# --- Helper Functions (mostly synchronous as they are simple UI updates or fast DB calls) ---
def update_model_dropdown(service_name: str):
    """
    Return available TTS models and the default selection for a given TTS service.
    
    Parameters:
        service_name (str): The name of the TTS service to query.
    
    Returns:
        dict: A dictionary with 'choices' as the list of available models and 'value' as the default model.
    """
    models = TTSManager.get_available_models(service_name)
    default_value = models[0] if models else None
    return {"choices": models, "value": default_value}

async def get_story_playback_data_async():
    """
    Asynchronously retrieves and formats the story history from the database for display in a Gradio Chatbot.
    
    Returns:
        list: A list of message dictionaries with 'role' and 'content' keys, formatted for Gradio's Chatbot component. If no history is found, returns a system message indicating the absence of story history.
    
    Raises:
        RuntimeError: If the database instance is not initialized.
    """
    # This DB read is usually fast. For very long stories, running in a thread avoids blocking the event loop.
    if db_instance is None:
        raise RuntimeError(
            "Database instance not initialized. Call launch_interface() first."
        )
    history_raw = await asyncio.to_thread(db_instance.get_story_history)
    chatbot_messages = []
    if not history_raw:
        return [{"role": "system", "content": "No story history found."}]
    for entry in history_raw:
        speaker = entry.get("speaker", "Unknown")
        text = entry.get("text", "")
        timestamp = entry.get("timestamp", "")
        formatted_text = f"_{timestamp}_ \n**{speaker}:** {text}"
        role = "user" if speaker.lower() == "narrator" else "assistant"
        chatbot_messages.append({"role": role, "content": formatted_text})
    return chatbot_messages

# --- Asynchronous Gradio Event Handlers ---

async def create_character_async(name, personality, goals, backstory, tts_service, tts_model, reference_audio_file, Actor_id, progress=gr.Progress(track_tqdm=True)):
    """
    Asynchronously creates or updates a character with the specified attributes, handling reference audio processing and token generation as needed.
    
    If the TTS service is "xttsv2" and a reference audio file is provided, the audio is saved to a configured directory with a sanitized filename. The character details are then saved to the database. If the Actor ID is not "Actor1", a token is generated for the actor. Progress updates are reported throughout the process.
    
    Parameters:
        name (str): The character's name.
        personality (str): Description of the character's personality.
        goals (str): The character's goals.
        backstory (str): The character's backstory.
        tts_service (str): The selected text-to-speech service.
        tts_model (str): The TTS model to use.
        reference_audio_file (file-like or None): Reference audio file for voice cloning (required for "xttsv2").
        Actor_id (str): The identifier for the actor.
        progress (gr.Progress, optional): Gradio progress tracker.
    
    Returns:
        str: Status message indicating success or error, and a token if generated.
    """
    if db_instance is None:
        raise RuntimeError("Database instance not initialized. Call launch_interface() first.")
    if client_manager_instance is None:
        raise RuntimeError("ClientManager instance not initialized. Call launch_interface() first.")
    progress(0, desc="Initializing character creation...")

    reference_audio_filename = None
    if reference_audio_file and tts_service == "xttsv2":
        progress(0.2, desc="Processing reference audio...")
        # Ensure the directory from config exists (config.py does this, but good practice)
        os.makedirs(REFERENCE_VOICES_AUDIO_PATH, exist_ok=True)
        sane_name = "".join(c if c.isalnum() else "_" for c in name)
        original_filename = reference_audio_file.name
        _, ext = os.path.splitext(original_filename)
        reference_audio_filename = f"{sane_name}_{Actor_id}_{os.urandom(4).hex()}{ext}"
        destination_path = os.path.join(REFERENCE_VOICES_AUDIO_PATH, reference_audio_filename)

        try:
            await asyncio.to_thread(shutil.copyfile, original_filename, destination_path)
            logger.info(f"Saved reference audio for character {name} ({Actor_id}) to: {destination_path}")
            if hasattr(progress, '__call__'):
                progress(0.5, desc="Reference audio saved.")
        except Exception as e:
            logger.error(f"Error saving reference audio for character {name} ({Actor_id}): {e}", exc_info=True)
            return f"Error saving reference audio: {e}"
    elif tts_service == "xttsv2" and not reference_audio_file:
        logger.warning(f"XTTS-v2 selected for {name} ({Actor_id}) but no reference audio file uploaded.")
        return "Error: XTTS-v2 selected but no reference audio file uploaded."

    if hasattr(progress, '__call__'):
        progress(0.7, desc="Saving character details to database...")
    await asyncio.to_thread(
        db_instance.save_character,
        name, personality, goals, backstory, tts_service, tts_model,
        reference_audio_filename, Actor_id,
        None  # llm_model is not provided by UI yet
    )

    token_msg_part = ""
    if Actor_id != "Actor1":
        # Token generation is fast (secrets.token_hex + DB write)
        token = await asyncio.to_thread(client_manager_instance.generate_token, Actor_id)
        if token:
            token_msg_part = f" Token for {Actor_id}: {token}"
            logger.info(f"Generated token for {Actor_id}.")

    if hasattr(progress, '__call__'):
        progress(1, desc="Character created!")
    logger.info(f"Character '{name}' for '{Actor_id}' created successfully. Token part: {token_msg_part}")
    return f"Character '{name}' for '{Actor_id}' created successfully.{token_msg_part}"


async def story_interface_async(audio_input_path, chaos_level_value, progress=gr.Progress(track_tqdm=True)):
    """
    Asynchronously processes a narration audio input to generate story narration and character dialogues.
    
    Parameters:
        audio_input_path: Path to the narration audio file.
        chaos_level_value: Value controlling the randomness or variability in story progression.
    
    Returns:
        narration: The generated narration text.
        character_texts: A dictionary mapping character names to their dialogue or responses. If an error occurs, returns an error message and empty outputs.
    """
    if csm_instance is None:
        raise RuntimeError("CSM instance not initialized. Call launch_interface() first.")
    if audio_input_path is None:
        return "No audio input. Record or upload narration.", {}, {}
    if hasattr(progress, '__call__'):
        progress(0, desc="Starting story processing...")
    try:
        narration, character_texts = await csm_instance.process_story(audio_input_path, chaos_level_value)
        if hasattr(progress, '__call__'):
            progress(1, desc="Story turn processed.")
        if not isinstance(character_texts, dict):
            character_texts = {"error": "Invalid character text format from CSM"}
        logger.info(f"Story processed with audio: {audio_input_path}, chaos: {chaos_level_value}.")
        return narration, character_texts
    except Exception as e:
        logger.error(f"Error in story_interface_async with audio {audio_input_path}: {e}", exc_info=True)
        if hasattr(progress, '__call__'):
            progress(1, desc="Error during story processing.")
        return f"Error: {e}", {}, {}

async def save_checkpoint_async(name_prefix, progress=gr.Progress(track_tqdm=True)):
    """
    Asynchronously saves a checkpoint with the specified name prefix and updates the checkpoint dropdown options.
    
    Parameters:
        name_prefix (str): Prefix to use for the checkpoint name.
    
    Returns:
        tuple: A status message and a dictionary containing updated checkpoint dropdown choices and the default value.
    """
    if checkpoint_manager is None:
        logger.error("CheckpointManager instance not initialized in save_checkpoint_async.")
        raise RuntimeError("CheckpointManager instance not initialized. Call launch_interface() first.")
    if hasattr(progress, '__call__'):
        progress(0, desc="Saving checkpoint...")
    logger.info(f"Attempting to save checkpoint with prefix: {name_prefix}")
    status, new_choices = await asyncio.to_thread(checkpoint_manager.save_checkpoint, name_prefix)
    if hasattr(progress, '__call__'):
        progress(1, desc="Checkpoint saved.")
    logger.info(f"Save checkpoint '{name_prefix}' result: {status}")
    return status, {"choices": new_choices, "value": new_choices[0] if new_choices else None}

async def load_checkpoint_async(checkpoint_name, progress=gr.Progress(track_tqdm=True)):
    """
    Asynchronously loads a specified checkpoint and refreshes the story playback data if successful.
    
    Parameters:
    	checkpoint_name (str): The name of the checkpoint to load.
    
    Returns:
    	tuple: A tuple containing the status message (str) and the updated story playback data (list).
    """
    if checkpoint_manager is None:
        logger.error("CheckpointManager instance not initialized in load_checkpoint_async.")
        raise RuntimeError("CheckpointManager instance not initialized. Call launch_interface() first.")
    if not checkpoint_name:
        logger.warning("Load checkpoint attempted without selecting a checkpoint name.")
        return "Please select a checkpoint to load.", []
    if hasattr(progress, '__call__'):
        progress(0, desc=f"Loading checkpoint '{checkpoint_name}'...")
    logger.info(f"Attempting to load checkpoint: {checkpoint_name}")
    status = await asyncio.to_thread(checkpoint_manager.load_checkpoint, checkpoint_name)
    new_story_data = []
    if status and "loaded" in status.lower() and "restart" not in status.lower():
        logger.info(f"Checkpoint '{checkpoint_name}' loaded successfully. Refreshing story history.")
        if hasattr(progress, '__call__'):
            progress(0.8, desc="Refreshing story history...")
        new_story_data = await get_story_playback_data_async()
    else:
        logger.warning(f"Checkpoint '{checkpoint_name}' load status: {status}. Story history not refreshed.")
    if hasattr(progress, '__call__'):
        progress(1, desc=f"Checkpoint '{checkpoint_name}' load attempt finished.")
    return status, new_story_data

async def export_story_async(export_format, progress=gr.Progress(track_tqdm=True)):
    """
    Asynchronously exports the current story in the specified format.
    
    Parameters:
        export_format (str): The format to export the story in (e.g., "json", "txt").
    
    Returns:
        status (str): The result message of the export operation.
        dict: A dictionary containing the exported filename and its visibility status.
    """
    if checkpoint_manager is None:
        logger.error("CheckpointManager instance not initialized in export_story_async.")
        raise RuntimeError("CheckpointManager instance not initialized. Call launch_interface() first.")
    if hasattr(progress, '__call__'):
        progress(0, desc=f"Exporting story as {export_format}...")
    logger.info(f"Attempting to export story as {export_format}.")
    status, filename = await asyncio.to_thread(checkpoint_manager.export_story, export_format)
    if hasattr(progress, '__call__'):
        progress(1, desc="Story export finished.")
    logger.info(f"Export story as {export_format} result: {status}, filename: {filename}")
    return status, {"value": filename if filename else "", "visible": bool(filename)}

async def get_env_vars_async():
    """
    Asynchronously retrieves the status of the `.env` file and loads environment variables with sensitive values masked.
    
    Returns:
        status (str): Status message about the `.env` file.
        vars_display_str (str): Formatted string of environment variables with sensitive values masked.
    """
    status = await asyncio.to_thread(env_manager.get_env_file_status)
    masked_vars = await asyncio.to_thread(env_manager.load_env_vars, mask_sensitive=True)

    vars_display_str = "\n".join([f"{k}={v}" for k, v in masked_vars.items()])
    if not vars_display_str:
        vars_display_str = "# No variables found or .env file does not exist."

    return status, vars_display_str

async def save_env_vars_async(new_vars_str: str, progress=gr.Progress()):
    """
    Asynchronously saves new or updated environment variables to the `.env` file and refreshes the displayed variables.
    
    Parameters:
        new_vars_str (str): The string containing new or updated environment variable definitions.
    
    Returns:
        status_msg (str): The result message from saving the `.env` file.
        new_status (str): The updated status of the `.env` file.
        new_vars_display (str): The refreshed display of environment variables with sensitive values masked.
    """
    progress(0, desc="Saving .env file...")
    logger.info("Attempting to save .env variables.")
    status_msg = await asyncio.to_thread(env_manager.save_env_vars, new_vars_str)
    progress(1, desc="Save complete.")
    logger.info(f".env file save result: {status_msg}")

    # After saving, refresh the display
    new_status, new_vars_display = await get_env_vars_async()

    return status_msg, new_status, new_vars_display

async def set_api_provider_async(selected_provider, progress=gr.Progress()):
    """
    Asynchronously updates the API provider in the environment variables and refreshes the displayed .env status.
    
    Parameters:
        selected_provider (str): The new API provider to set.
    
    Returns:
        status_msg (str): Status message indicating the result of the update.
        new_status (str): Updated .env file status.
        new_vars_display (str): Masked string representation of environment variables.
        selected_provider (str): The provider that was set.
    """
    progress(0, desc="Updating API provider...")
    new_var = f"API_PROVIDER={selected_provider}"
    logger.info(f"Attempting to set API_PROVIDER to: {selected_provider}")
    status_msg = await asyncio.to_thread(env_manager.save_env_vars, new_var)
    progress(1, desc="Provider updated.")
    logger.info(f"API_PROVIDER update result: {status_msg}")
    # After saving, refresh the display
    new_status, new_vars_display = await get_env_vars_async()
    return status_msg, new_status, new_vars_display, selected_provider

async def restart_server_async(progress=gr.Progress()):
    """
    Asynchronously restarts the server process by re-executing the current Python interpreter.
    
    Returns:
        str: Status message indicating the server is restarting.
    """
    progress(0, desc="Restarting server...")
    logger.info("Server restart requested via Gradio UI.")
    await asyncio.sleep(1) # Give time for the message to show in UI
    progress(1, desc="Server restarting now.")
    # Attempt to restart the current process
    logger.info(f"Executing os.execv: {sys.executable} {sys.argv}")
    os.execv(sys.executable, [sys.executable] + sys.argv)
    return "Server is restarting..." # This line might not be reached if execv is successful

# --- Gradio UI Launch ---
def launch_interface():
    """
    Launches the asynchronous Gradio web interface for the Dream Weaver storytelling and character management system.
    
    Initializes all core backend components and defines a multi-tabbed UI for character creation, story narration, playback/history, checkpoint management, environment variable editing, and configuration. Integrates asynchronous event handlers for non-blocking operations, dynamic UI updates, and progress reporting. The interface supports audio narration, TTS model selection, checkpointing, story export, API key management, and live configuration editing, with all changes reflected in real time. The server is launched on all interfaces at port 7860 with async queueing enabled.
    """
    global db_instance, client_manager_instance, csm_instance, checkpoint_manager, env_manager_instance
    db_instance = Database(DB_PATH)
    client_manager_instance = ClientManager(db_instance)
    csm_instance = CSM()
    checkpoint_manager = CheckpointManager()
    # env_manager is a module of functions, no instance needed, but keeping pattern
    env_manager_instance = env_manager

    import gradio.themes as themes
    with gr.Blocks(theme=themes.Soft(primary_hue=themes.colors.indigo, secondary_hue=themes.colors.blue)) as demo:
        gr.Markdown("# Dream Weaver Interface")

        with gr.Tabs():
            with gr.TabItem("Character Management"):
                # ... (Character Management UI definition - use create_character_async) ...
                gr.Markdown("## Create or Update Characters")
                with gr.Row():
                    with gr.Column(scale=2):
                        char_name = gr.Textbox(label="Character Name", placeholder="E.g., 'Elara'")
                        char_Actor_id = gr.Dropdown(["Actor1"] + [f"Actor{i}" for i in range(2, 11)], label="Assign to Actor ID", value="Actor1")
                        char_tts_service = gr.Dropdown(TTSManager.list_services(), label="TTS Service")
                        char_tts_model = gr.Dropdown([], label="TTS Model")
                        char_ref_audio = gr.File(label="Reference Audio (XTTSv2)", type="filepath", visible=False)
                    with gr.Column(scale=3):
                        char_personality = gr.Textbox(label="Personality", lines=2)
                        char_goals = gr.Textbox(label="Goals", lines=2)
                        char_backstory = gr.Textbox(label="Backstory", lines=3)

                create_char_btn = gr.Button("Save Character", variant="primary")
                char_creation_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("Client Configuration Download", open=False):
                    client_config_server_url = gr.Textbox(label="Server URL for Client", placeholder="E.g., http://192.168.1.100:8000 or http://your.domain.com:8000")
                    download_client_config_btn = gr.Button("Download Client .env File")
                    # Hidden File component for download trick
                    client_config_file_download = gr.File(label="Download Link", visible=False, interactive=False)
                    client_config_download_status = gr.Textbox(label="Download Status", interactive=False, visible=False)


                char_tts_service.change(fn=update_model_dropdown, inputs=char_tts_service, outputs=char_tts_model)
                char_tts_service.change(lambda service: {"visible": (service == "xttsv2")}, inputs=char_tts_service, outputs=char_ref_audio)

                create_char_btn.click(
                    create_character_async,
                    inputs=[char_name, char_personality, char_goals, char_backstory, char_tts_service, char_tts_model, char_ref_audio, char_Actor_id],
                    outputs=char_creation_status
                )

                async def handle_download_client_config(actor_id, server_url_for_client, progress=gr.Progress()):
                    """
                    Prepares and triggers the download of a client configuration .env file.
                    """
                    if not actor_id or actor_id == "Actor1":
                        logger.warning(f"Client config download requested for invalid actor_id: {actor_id}")
                        return {client_config_download_status: gr.update(value="Select a valid Client Actor ID (not Actor1).", visible=True),
                                client_config_file_download: gr.update(visible=False)}
                    if not server_url_for_client:
                        logger.warning(f"Client config download requested for actor_id: {actor_id} without server_url.")
                        return {client_config_download_status: gr.update(value="Please enter the Server URL for the client.", visible=True),
                                client_config_file_download: gr.update(visible=False)}

                    progress(0, desc="Preparing download link...")
                    logger.info(f"Preparing client config download for Actor_id: {actor_id} with server_url: {server_url_for_client}")

                    # Construct the download URL for the FastAPI endpoint
                    # Ensure server_url_for_client is URL-encoded if it contains special characters, though Gradio might handle this.
                    # For direct URL construction, it's safer.
                    from urllib.parse import quote
                    encoded_server_url = quote(server_url_for_client, safe=':/')

                    # Assuming Gradio server is running on http://localhost:7860 and FastAPI on http://localhost:8000 (or as configured)
                    # The download URL needs to point to the FastAPI endpoint.
                    # We need the actual base URL of the FastAPI server as seen by the user's browser.
                    # This is tricky if Gradio and FastAPI are on different ports or hosts from browser's perspective.
                    # For now, let's assume they are on the same host and FastAPI is on port 8000.
                    # This might need to be configurable or determined more robustly.
                    # For now, let's hardcode the relative path to the API, assuming Gradio proxies or they are on same host.
                    # A better way would be to get this from a config or request object if possible.
                    # Simplest for now: assume FastAPI is reachable from where Gradio UI is served.
                    # The API endpoint itself is on the FastAPI server, not Gradio.
                    # So the URL should be like "http://actual_fastapi_host:fastapi_port/download_client_config/..."
                    # If Gradio is served from say 0.0.0.0:7860 and API from 0.0.0.0:8000 (defaults)
                    # the user's browser needs to be able to hit 0.0.0.0:8000.
                    # Let's assume the user knows the correct FastAPI base URL.
                    # The `server_url_for_client` is what the *client* will use.
                    # The download link itself is for the *browser* interacting with Gradio/FastAPI.

                    # The FastAPI app is available at the root of its own server (e.g. http://localhost:8000)
                    # This needs to be the actual URL to the FastAPI server.
                    # For testing locally, if Gradio is 7860 and FastAPI is 8000:
                    fastapi_base_url = "http://localhost:8000" # This should ideally be configurable or derived

                    download_url = f"{fastapi_base_url}/download_client_config/{actor_id}?server_url={encoded_server_url}"

                    progress(1, desc="Link generated. Starting download...")

                    # Use gr.File to trigger download. Value is the URL.
                    # Make it visible briefly then hide again.
                    # This is a bit of a hack. A direct gr.DownloadButton would be ideal if it could be dynamically targeted.
                    logger.info(f"Triggering download for {actor_id} with URL: {download_url}")
                    return {
                        client_config_download_status: gr.update(value=f"Preparing download for {actor_id}...", visible=True),
                        client_config_file_download: gr.update(value=download_url, visible=True) # This should trigger download
                    }

                async def hide_download_link_after_trigger():
                    logger.debug("Hiding client config download link elements after delay.")
                    await asyncio.sleep(2) # Keep it visible for a couple of seconds
                    return {
                        client_config_file_download: gr.update(visible=False),
                        client_config_download_status: gr.update(visible=False)
                    }

                download_client_config_btn.click(
                    handle_download_client_config,
                    inputs=[char_Actor_id, client_config_server_url],
                    outputs=[client_config_download_status, client_config_file_download]
                ).then(
                    hide_download_link_after_trigger,
                    inputs=[],
                    outputs=[client_config_file_download, client_config_download_status]
                )


            with gr.TabItem("Story Progression"):
                # ... (Story Progression UI definition - use story_interface_async) ...
                gr.Markdown("## Narrate the Story")
                with gr.Row():
                    with gr.Column(scale=2):
                        story_audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Narration")
                        story_chaos_slider = gr.Slider(minimum=0, maximum=10, value=1, step=1, label="Chaos Level")
                        process_story_btn = gr.Button("Process Narration", variant="primary")
                    with gr.Column(scale=3):
                        narration_output_text = gr.Textbox(label="Narrator's Words", lines=3, interactive=True)
                        save_correction_btn = gr.Button("Save Correction", variant="secondary")
                        correction_status = gr.Textbox(label="Correction Status", interactive=False)
                        character_responses_json = gr.JSON(label="Character Dialogues") # Changed from Textbox

                process_story_btn.click(
                    story_interface_async,
                    inputs=[story_audio_input, story_chaos_slider],
                    outputs=[narration_output_text, character_responses_json]
                )

                # --- Correction Handler ---
                async def save_correction_async(correction_text, progress=gr.Progress(track_tqdm=True)):
                    """
                    Asynchronously updates the last narration text with a correction.
                    
                    Parameters:
                        correction_text (str): The corrected narration text to save.
                    
                    Returns:
                        str: Status message indicating whether the correction was saved, no narrator entry was found, or the character/story manager (CSM) is not initialized.
                    """
                    progress(0, desc="Saving correction...")
                    # Persist correction to DB
                    if csm_instance is not None:
                        logger.info(f"Attempting to save narration correction: {correction_text[:100]}...") # Log first 100 chars
                        updated = await asyncio.to_thread(csm_instance.update_last_narration_text, correction_text)
                        if updated:
                            progress(1, desc="Correction saved.")
                            logger.info("Narration correction saved successfully.")
                            return "Correction saved and persisted!"
                        else:
                            progress(1, desc="No narrator entry found.")
                            logger.warning("No narrator entry found to update for correction.")
                            return "No narrator entry found to update."
                    progress(1, desc="CSM not initialized.")
                    logger.error("CSM instance not initialized in save_correction_async.")
                    return "CSM not initialized."
                save_correction_btn.click(save_correction_async, inputs=[narration_output_text], outputs=[correction_status])

            with gr.TabItem("Story Playback & History"):
                # ... (Story Playback UI - get_story_playback_data_sync is likely fine for initial load and refresh unless very slow) ...
                gr.Markdown("## Review Story History")
                story_playback_chatbot = gr.Chatbot(
                    label="Full Story Log",
                    height=600,
                    show_copy_button=True,
                    type="messages",
                )
                refresh_story_btn = gr.Button("Refresh Story History")

                # Making this async too for consistency, though DB reads are usually fast.
                async def refresh_story_async_wrapper(progress=gr.Progress()):
                    """
                    Asynchronously fetches and returns the latest story playback data, updating progress status during the operation.
                    
                    Returns:
                        data (list): List of formatted story history messages for display.
                    """
                    progress(0, desc="Fetching history...")
                    data = await get_story_playback_data_async()
                    progress(1, desc="History loaded.")
                    return data

                refresh_story_btn.click(refresh_story_async_wrapper, inputs=[], outputs=[story_playback_chatbot])
                demo.load(refresh_story_async_wrapper, inputs=[], outputs=[story_playback_chatbot]) # Initial load

            with gr.TabItem("System & Data Management"):
                # ... (Checkpoints & Export UI - use ..._async handlers) ...
                gr.Markdown("## Checkpoints & Export")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Checkpoints")
                        chkpt_name_prefix = gr.Textbox(label="Checkpoint Name Prefix", placeholder="E.g., 'AfterTheHeist'")
                        save_chkpt_btn = gr.Button("Save Checkpoint")

                        current_checkpoints = checkpoint_manager.list_checkpoints() # Sync, fast
                        chkpt_dropdown = gr.Dropdown(choices=current_checkpoints, label="Load Checkpoint",
                                                     value=current_checkpoints[0] if current_checkpoints else None)
                        load_chkpt_btn = gr.Button("Load Selected Checkpoint")
                        chkpt_status = gr.Textbox(label="Checkpoint Status", interactive=False)

                        save_chkpt_btn.click(
                            save_checkpoint_async,
                            inputs=[chkpt_name_prefix],
                            outputs=[chkpt_status, chkpt_dropdown]
                        )
                        load_chkpt_btn.click(
                            load_checkpoint_async,
                            inputs=[chkpt_dropdown],
                            outputs=[chkpt_status, story_playback_chatbot]
                        )
                    with gr.Column():
                        gr.Markdown("### Export Story")
                        export_format_radio = gr.Radio(["text", "json"], label="Export Format", value="text")
                        export_story_btn = gr.Button("Export Full Story")
                        export_status_text = gr.Textbox(label="Export Status")
                        export_filename_display = gr.Textbox(label="Exported To")

                        export_story_btn.click(
                            export_story_async,
                            inputs=[export_format_radio],
                            outputs=[export_status_text, export_filename_display]
                        )

            with gr.TabItem("API Keys & .env"):
                gr.Markdown("## Manage Environment Variables (.env)")
                gr.Markdown(
                    "Add or update environment variables here, such as `HUGGING_FACE_HUB_TOKEN` or `NASA_API_KEY`. "
                    "Values for keys containing 'TOKEN', 'KEY', or 'SECRET' will be masked for security. "
                    "**A server restart is required for any changes to take effect.**"
                )

                env_status_text = gr.Textbox(label=".env File Status", interactive=False)

                # --- API Provider Dropdown ---
                api_providers = ["huggingface", "openai", "google", "custom"]
                provider_token_vars = {
                    "huggingface": ("HUGGING_FACE_HUB_TOKEN", "Hugging Face Token", "hf_..."),
                    "openai": ("OPENAI_API_KEY", "OpenAI API Key", "sk-..."),
                    "google": ("GOOGLE_API_KEY", "Google API Key", "AIza..."),
                    "custom": ("CUSTOM_API_TOKEN", "Custom API Token", "token...")
                }
                def get_current_provider():
                    """
                    Return the currently selected API provider from environment variables, or the default provider if not set.
                    """
                    env_vars = env_manager.load_env_vars(mask_sensitive=False)
                    return env_vars.get("API_PROVIDER", api_providers[0])
                api_provider_dropdown = gr.Dropdown(api_providers, label="API Provider", value=get_current_provider())
                api_provider_status = gr.Textbox(label="API Provider Status", interactive=False)

                # Dynamic token input
                def get_token_field(provider):
                    """
                    Return the configuration for the API token input field based on the selected provider.
                    
                    Parameters:
                        provider (str): The name of the API provider.
                    
                    Returns:
                        dict: A dictionary specifying the visibility, label, placeholder, and initial value for the token input field.
                    """
                    var, label, placeholder = provider_token_vars.get(provider, ("API_TOKEN", "API Token", "token..."))
                    return {"visible": True, "label": label, "placeholder": placeholder, "value": ""}
                def hide_token_field():
                    """
                    Hide the token input field in the UI by setting its visibility to False.
                    
                    Returns:
                    	dict: A dictionary indicating the token field should be hidden.
                    """
                    return {"visible": False}
                token_input = gr.Textbox(label="API Token", visible=True, placeholder="token...")

                # Show/hide and relabel token input on provider change
                def update_token_field(provider):
                    """
                    Return UI field configuration for the API token input based on the selected provider.
                    
                    If the provider is recognized, returns a dictionary specifying the field's visibility, label, placeholder, and an empty value. Otherwise, returns a dictionary hiding the field.
                    """
                    if provider in provider_token_vars:
                        var, label, placeholder = provider_token_vars[provider]
                        return {"visible": True, "label": label, "placeholder": placeholder, "value": ""}
                    return {"visible": False}
                api_provider_dropdown.change(update_token_field, inputs=[api_provider_dropdown], outputs=[token_input])

                # Save token handler
                async def save_token_async(provider, token, progress=gr.Progress()):
                    """
                    Asynchronously saves an API token for the selected provider to the environment variables.
                    
                    Parameters:
                        provider (str): The name of the API provider.
                        token (str): The API token to save.
                    
                    Returns:
                        status_msg (str): Status message indicating the result of the save operation.
                        new_status (str): Updated status of the environment variables file.
                        new_vars_display (str): Formatted string of environment variables with sensitive values masked.
                        (str): An empty string (reserved for UI compatibility).
                    """
                    progress(0, desc="Saving token...")
                    var, _, _ = provider_token_vars.get(provider, ("API_TOKEN", "API Token", "token..."))
                    new_var = f"{var}={token}" # Token itself is not logged for security
                    logger.info(f"Attempting to save token for provider: {provider} (variable: {var})")
                    status_msg = await asyncio.to_thread(env_manager.save_env_vars, new_var)
                    progress(1, desc="Token saved.")
                    logger.info(f"Save token for {provider} result: {status_msg}")
                    new_status, new_vars_display = await get_env_vars_async()
                    return status_msg, new_status, new_vars_display, ""
                save_token_btn = gr.Button("Save API Token", variant="primary")
                save_token_status = gr.Textbox(label="Token Save Status", interactive=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Current Variables (Masked)")
                        current_env_vars_display = gr.Textbox(
                            label="Current .env Content",
                            lines=10,
                            interactive=False,
                            placeholder="# .env file content will be shown here."
                        )
                    with gr.Column():
                        gr.Markdown("### Add or Update Variables")
                        new_env_vars_input = gr.Textbox(
                            label="New or Updated Variables (one per line)",
                            lines=10,
                            placeholder="HUGGING_FACE_HUB_TOKEN=hf_...\nANOTHER_KEY=some_value"
                        )

                save_env_btn = gr.Button("Save to .env", variant="primary")
                save_status_text = gr.Textbox(label="Save Status", interactive=False)
                restart_required_text = gr.Markdown("**Restart required for changes to take effect.**", visible=False)
                restart_btn = gr.Button("Restart Server", variant="stop", visible=False)

                # Event Handlers for this tab
                def show_restart():
                    """
                    Show the server restart message and button by setting their visibility to True.
                    
                    Returns:
                        tuple: Two dictionaries indicating visibility for the restart message and button.
                    """
                    return {"visible": True}, {"visible": True}
                demo.load(get_env_vars_async, inputs=[], outputs=[env_status_text, current_env_vars_display])
                save_env_btn.click(save_env_vars_async, inputs=[new_env_vars_input], outputs=[save_status_text, env_status_text, current_env_vars_display])
                save_env_btn.click(show_restart, inputs=[], outputs=[restart_required_text, restart_btn])
                api_provider_dropdown.change(set_api_provider_async, inputs=[api_provider_dropdown], outputs=[api_provider_status, env_status_text, current_env_vars_display, api_provider_dropdown])
                save_token_btn.click(save_token_async, inputs=[api_provider_dropdown, token_input], outputs=[save_token_status, env_status_text, current_env_vars_display, token_input])
                save_token_btn.click(show_restart, inputs=[], outputs=[restart_required_text, restart_btn])
                restart_btn.click(restart_server_async, inputs=[], outputs=[])

            with gr.TabItem("Config & Model Options"):
                gr.Markdown("## Edit Config & Model Options")
                from .config import EDITABLE_CONFIG_OPTIONS
                config_keys = list(EDITABLE_CONFIG_OPTIONS.keys())
                config_values = [str(EDITABLE_CONFIG_OPTIONS[k]) for k in config_keys]
                config_inputs = [gr.Textbox(label=k, value=v) for k, v in zip(config_keys, config_values)]
                save_config_btn = gr.Button("Save Config Changes", variant="primary")
                config_status = gr.Textbox(label="Config Save Status", interactive=False)

                async def save_config_async(*new_values):
                    # Save new config values to .env or another persistent store
                    """
                    Asynchronously saves updated configuration values to the environment variable store.
                    
                    Parameters:
                    	new_values: New configuration values corresponding to predefined config keys.
                    
                    Returns:
                    	status_msg (str): Status message indicating the result of the save operation.
                    """
                    lines = []
                    for k, v in zip(config_keys, new_values):
                        lines.append(f"{k}={v}")
                    logger.info(f"Attempting to save config changes to .env: {lines}")
                    # Save to .env for persistence
                    status_msg = await asyncio.to_thread(env_manager.save_env_vars, "\n".join(lines))
                    logger.info(f"Save config changes result: {status_msg}")
                    return status_msg
                save_config_btn.click(save_config_async, inputs=config_inputs, outputs=[config_status])

        gr.Markdown("---")
        gr.Markdown("View [Server Dashboard](/dashboard) (Server Perf & Client Status)")

    # Launch the Gradio app
    # For multi-worker Uvicorn, ensure global instances (db_instance, csm_instance etc.) are handled safely.
    # This typically means initializing them within the FastAPI app's lifespan events or using dependency injection.
    # For now, assuming single worker or careful management.
    demo.queue().launch(server_name="0.0.0.0", server_port=7860) # .queue() is important for async handlers and progress

if __name__ == "__main__":
    print("Launching Gradio interface directly (SERVER/src/gradio_interface.py)...")
    # Ensure server config.py creates directories (it does on import)
    from .config import DB_PATH, REFERENCE_VOICES_AUDIO_PATH
    launch_interface()
