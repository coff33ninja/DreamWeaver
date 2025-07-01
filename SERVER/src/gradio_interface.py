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
    Return available TTS model choices and default selection for a given service.
    
    Parameters:
        service_name (str): The name of the TTS service for which to fetch available models.
    
    Returns:
        dict: A dictionary with keys 'choices' (list of model names) and 'value' (default model or None).
    """
    models = TTSManager.get_available_models(service_name)
    default_value = models[0] if models else None
    return {"choices": models, "value": default_value}

async def get_story_playback_data_async():
    """
    Asynchronously retrieves and formats the story history from the database for display in a Gradio Chatbot.
    
    Returns:
        list[dict]: A list of message dictionaries with 'role' and 'content' keys, suitable for Gradio Chatbot display. If no history is found, returns a single system message.
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
    Asynchronously creates or updates a character with the specified attributes and optional reference audio.
    
    If the TTS service is "xttsv2" and a reference audio file is provided, the audio is saved to the configured directory. The character details are then saved to the database. For non-default actors, a token is generated and returned in the success message. Progress updates are provided throughout the operation.
    
    Parameters:
        name (str): The character's name.
        personality (str): Description of the character's personality.
        goals (str): The character's goals or motivations.
        backstory (str): The character's backstory.
        tts_service (str): The selected text-to-speech service.
        tts_model (str): The TTS model to use.
        reference_audio_file (file-like or None): Reference audio file for voice cloning (required for "xttsv2").
        Actor_id (str): The unique identifier for the actor.
    
    Returns:
        str: Status message indicating success or error details.
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
            print(f"Saved reference audio to: {destination_path}")
            if hasattr(progress, '__call__'):
                progress(0.5, desc="Reference audio saved.")
        except Exception as e:
            print(f"Error saving reference audio: {e}")
            return f"Error saving reference audio: {e}"
    elif tts_service == "xttsv2" and not reference_audio_file:
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

    if hasattr(progress, '__call__'):
        progress(1, desc="Character created!")
    return f"Character '{name}' for '{Actor_id}' created successfully.{token_msg_part}"


async def story_interface_async(audio_input_path, chaos_level_value, progress=gr.Progress(track_tqdm=True)):
    """
    Processes a narration audio file asynchronously to generate story narration and character dialogues.
    
    Parameters:
    	audio_input_path: Path to the narration audio file.
    	chaos_level_value: Value controlling the randomness or creativity of the story generation.
    
    Returns:
    	narration: The generated narration text.
    	character_texts: A dictionary containing character dialogue outputs. If an error occurs, returns an error message and empty outputs.
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
        return narration, character_texts
    except Exception as e:
        print(f"Error in story_interface_async: {e}")
        if hasattr(progress, '__call__'):
            progress(1, desc="Error during story processing.")
        return f"Error: {e}", {}, {}

async def save_checkpoint_async(name_prefix, progress=gr.Progress(track_tqdm=True)):
    """
    Asynchronously saves a new checkpoint with the specified name prefix.
    
    Parameters:
        name_prefix (str): Prefix to use for the checkpoint name.
    
    Returns:
        tuple: A status message and a dictionary containing updated checkpoint dropdown choices and the default value.
    """
    if checkpoint_manager is None:
        raise RuntimeError("CheckpointManager instance not initialized. Call launch_interface() first.")
    if hasattr(progress, '__call__'):
        progress(0, desc="Saving checkpoint...")
    status, new_choices = await asyncio.to_thread(checkpoint_manager.save_checkpoint, name_prefix)
    if hasattr(progress, '__call__'):
        progress(1, desc="Checkpoint saved.")
    return status, {"choices": new_choices, "value": new_choices[0] if new_choices else None}

async def load_checkpoint_async(checkpoint_name, progress=gr.Progress(track_tqdm=True)):
    """
    Asynchronously loads a specified checkpoint and refreshes story playback data if successful.
    
    Parameters:
    	checkpoint_name (str): The name of the checkpoint to load.
    
    Returns:
    	status (str): The result message from the checkpoint load operation.
    	new_story_data (list): Updated story playback data if the checkpoint was loaded successfully; otherwise, an empty list.
    """
    if checkpoint_manager is None:
        raise RuntimeError("CheckpointManager instance not initialized. Call launch_interface() first.")
    if not checkpoint_name:
        return "Please select a checkpoint to load.", []
    if hasattr(progress, '__call__'):
        progress(0, desc=f"Loading checkpoint '{checkpoint_name}'...")
    status = await asyncio.to_thread(checkpoint_manager.load_checkpoint, checkpoint_name)
    new_story_data = []
    if status and "loaded" in status.lower() and "restart" not in status.lower():
        if hasattr(progress, '__call__'):
            progress(0.8, desc="Refreshing story history...")
        new_story_data = await get_story_playback_data_async()
    if hasattr(progress, '__call__'):
        progress(1, desc=f"Checkpoint '{checkpoint_name}' load attempt finished.")
    return status, new_story_data

async def export_story_async(export_format, progress=gr.Progress(track_tqdm=True)):
    """
    Asynchronously exports the current story in the specified format.
    
    Parameters:
        export_format (str): The desired export format (e.g., "json", "txt").
    
    Returns:
        tuple: A status message and a dictionary containing the exported filename and its visibility flag.
    """
    if checkpoint_manager is None:
        raise RuntimeError("CheckpointManager instance not initialized. Call launch_interface() first.")
    if hasattr(progress, '__call__'):
        progress(0, desc=f"Exporting story as {export_format}...")
    status, filename = await asyncio.to_thread(checkpoint_manager.export_story, export_format)
    if hasattr(progress, '__call__'):
        progress(1, desc="Story export finished.")
    return status, {"value": filename if filename else "", "visible": bool(filename)}

async def get_env_vars_async():
    """
    Asynchronously retrieves the status and masked contents of the `.env` file.
    
    Returns:
        status (str): The status message indicating the presence or state of the `.env` file.
        vars_display_str (str): A formatted string listing environment variables with sensitive values masked, or a message if no variables are found.
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
        new_vars_str (str): String containing the new or updated environment variable definitions.
    
    Returns:
        status_msg (str): Status message indicating the result of the save operation.
        new_status (str): Updated status of the `.env` file.
        new_vars_display (str): Formatted string of the current environment variables.
    """
    progress(0, desc="Saving .env file...")
    status_msg = await asyncio.to_thread(env_manager.save_env_vars, new_vars_str)
    progress(1, desc="Save complete.")

    # After saving, refresh the display
    new_status, new_vars_display = await get_env_vars_async()

    return status_msg, new_status, new_vars_display

async def set_api_provider_async(selected_provider, progress=gr.Progress()):
    """
    Asynchronously updates the API provider in the environment file and refreshes the displayed environment variables.
    
    Parameters:
        selected_provider (str): The new API provider to set.
    
    Returns:
        status_msg (str): Status message from the save operation.
        new_status (str): Updated status of the environment file.
        new_vars_display (str): Masked display of current environment variables.
        selected_provider (str): The provider that was set.
    """
    progress(0, desc="Updating API provider...")
    new_var = f"API_PROVIDER={selected_provider}"
    status_msg = await asyncio.to_thread(env_manager.save_env_vars, new_var)
    progress(1, desc="Provider updated.")
    # After saving, refresh the display
    new_status, new_vars_display = await get_env_vars_async()
    return status_msg, new_status, new_vars_display, selected_provider

async def restart_server_async(progress=gr.Progress()):
    """
    Asynchronously restarts the server process after a brief delay.
    
    Returns:
        str: Status message indicating the server is restarting.
    """
    progress(0, desc="Restarting server...")
    await asyncio.sleep(1)
    progress(1, desc="Server restarting now.")
    # Attempt to restart the current process
    os.execv(sys.executable, [sys.executable] + sys.argv)
    return "Server is restarting..."

# --- Gradio UI Launch ---
def launch_interface():
    """
    Launches the Dream Weaver Gradio web interface, initializing all core backend modules and defining the complete multi-tab asynchronous UI for character management, story narration, playback, checkpoints, environment variables, and configuration.
    
    This function sets up global instances for database, client management, story processing, checkpoint handling, and environment variable management. It constructs a Gradio Blocks interface with dedicated tabs for character creation, story progression, playback/history, system management (checkpoints and export), API key and environment variable management, and configuration editing. Each tab is wired to asynchronous event handlers for responsive, non-blocking user interactions, including progress tracking and dynamic UI updates.
    
    The interface supports audio narration, TTS model selection, reference audio upload, checkpoint save/load, story export, environment variable editing with masking for sensitive keys, API provider selection, server restart, and live configuration changes. The app is launched with async queueing enabled and listens on all interfaces at port 7860.
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

                char_tts_service.change(fn=update_model_dropdown, inputs=char_tts_service, outputs=char_tts_model)
                char_tts_service.change(lambda service: {"visible": (service == "xttsv2")}, inputs=char_tts_service, outputs=char_ref_audio)

                create_char_btn.click(
                    create_character_async,
                    inputs=[char_name, char_personality, char_goals, char_backstory, char_tts_service, char_tts_model, char_ref_audio, char_Actor_id],
                    outputs=char_creation_status
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
                    Update the last narrator entry in the story with the provided correction text.
                    
                    Parameters:
                    	correction_text (str): The corrected narration text to persist.
                    
                    Returns:
                    	str: Status message indicating whether the correction was saved, no narrator entry was found, or the CSM instance is uninitialized.
                    """
                    progress(0, desc="Saving correction...")
                    # Persist correction to DB
                    if csm_instance is not None:
                        updated = await asyncio.to_thread(csm_instance.update_last_narration_text, correction_text)
                        if updated:
                            progress(1, desc="Correction saved.")
                            return "Correction saved and persisted!"
                        else:
                            progress(1, desc="No narrator entry found.")
                            return "No narrator entry found to update."
                    progress(1, desc="CSM not initialized.")
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
                    Asynchronously refreshes and returns the latest story playback history for display in the UI.
                    
                    Parameters:
                        progress: Gradio progress tracker for UI feedback.
                    
                    Returns:
                        data (list): List of message dictionaries representing the story playback history.
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
                    Return the current API provider from environment variables, or the default provider if not set.
                    """
                    env_vars = env_manager.load_env_vars(mask_sensitive=False)
                    return env_vars.get("API_PROVIDER", api_providers[0])
                api_provider_dropdown = gr.Dropdown(api_providers, label="API Provider", value=get_current_provider())
                api_provider_status = gr.Textbox(label="API Provider Status", interactive=False)

                # Dynamic token input
                def get_token_field(provider):
                    """
                    Return a dictionary describing the token input field configuration for the specified API provider.
                    
                    Parameters:
                    	provider (str): The name of the API provider.
                    
                    Returns:
                    	dict: A dictionary with keys 'visible', 'label', 'placeholder', and 'value' for configuring the token input field.
                    """
                    var, label, placeholder = provider_token_vars.get(provider, ("API_TOKEN", "API Token", "token..."))
                    return {"visible": True, "label": label, "placeholder": placeholder, "value": ""}
                def hide_token_field():
                    """
                    Hide the token input field in the UI by setting its visibility to False.
                    
                    Returns:
                        dict: A dictionary indicating the token field should be hidden (`{"visible": False}`).
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
                    Asynchronously saves an API token for the selected provider to environment variables.
                    
                    Parameters:
                        provider (str): The name of the API provider.
                        token (str): The API token to save.
                    
                    Returns:
                        status_msg (str): Status message indicating the result of the save operation.
                        new_status (str): Updated status of the environment variables.
                        new_vars_display (str): Formatted string displaying the updated environment variables.
                        (str): An empty string placeholder for UI compatibility.
                    """
                    progress(0, desc="Saving token...")
                    var, _, _ = provider_token_vars.get(provider, ("API_TOKEN", "API Token", "token..."))
                    new_var = f"{var}={token}"
                    status_msg = await asyncio.to_thread(env_manager.save_env_vars, new_var)
                    progress(1, desc="Token saved.")
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
                    Return UI visibility settings to show the server restart button and related elements.
                    
                    Returns:
                        tuple: Two dictionaries indicating visibility for UI components.
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
                    Asynchronously saves updated configuration values to persistent storage.
                    
                    Parameters:
                        *new_values: New values for each configuration key, in order.
                    
                    Returns:
                        status_msg (str): Status message indicating the result of the save operation.
                    """
                    lines = []
                    for k, v in zip(config_keys, new_values):
                        lines.append(f"{k}={v}")
                    # Save to .env for persistence
                    status_msg = await asyncio.to_thread(env_manager.save_env_vars, "\n".join(lines))
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
