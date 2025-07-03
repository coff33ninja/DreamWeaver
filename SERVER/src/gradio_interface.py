import gradio as gr
from .csm import CSM
from .database import Database
from .tts_manager import TTSManager  # Server's TTSManager
from .client_manager import ClientManager
from .checkpoint_manager import CheckpointManager
from . import env_manager
from .config import DB_PATH, REFERENCE_VOICES_AUDIO_PATH
from .hardware import Hardware

# Import the global connection_manager
from .websocket_manager import connection_manager as global_ws_connection_manager

import shutil
import os
import asyncio
import sys
import logging
from urllib.parse import quote


logger = logging.getLogger("dreamweaver_server")

# --- Instances ---
db_instance = None
client_manager_instance = None
csm_instance = None
checkpoint_manager = None
env_manager_instance = None


# --- Helper Functions ---
def update_model_dropdown(service_name: str):
    models = TTSManager.get_available_models(service_name)
    default_value = models[0] if models else None
    return gr.update(choices=models, value=default_value)  # Use gr.update


async def get_story_playback_data_async():
    if db_instance is None:
        raise RuntimeError("Database instance not initialized.")
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


async def get_adapter_ips_async():
    # Use a thread to avoid blocking
    return Hardware.get_adapter_ip_addresses()


# --- Async Handlers (Existing ones condensed for brevity in this example) ---
async def create_character_async(
    name,
    personality,
    goals,
    backstory,
    tts_service,
    tts_model,
    reference_audio_file,
    Actor_id,
    progress=gr.Progress(track_tqdm=True),
):
    # ... (implementation as before) ...
    if db_instance is None:
        raise RuntimeError("Database instance not initialized.")
    if client_manager_instance is None:
        raise RuntimeError("ClientManager instance not initialized.")
    progress(0, desc="Initializing character creation...")
    reference_audio_filename = None
    if tts_service == "xttsv2" and reference_audio_file:
        progress(0.2, desc="Processing reference audio...")
        os.makedirs(REFERENCE_VOICES_AUDIO_PATH, exist_ok=True)
        sane_name = "".join(c if c.isalnum() else "_" for c in name)
        original_filename = reference_audio_file.name
        _, ext = os.path.splitext(original_filename)
        reference_audio_filename = f"{sane_name}_{Actor_id}_{os.urandom(4).hex()}{ext}"
        destination_path = os.path.join(
            REFERENCE_VOICES_AUDIO_PATH, reference_audio_filename
        )
        try:
            await asyncio.to_thread(
                shutil.copyfile, original_filename, destination_path
            )
            logger.info(
                f"Saved reference audio for character {name} ({Actor_id}) to: {destination_path}"
            )
            progress(0.5, desc="Reference audio saved.")
        except Exception as e:
            logger.error(
                f"Error saving reference audio for character {name} ({Actor_id}): {e}",
                exc_info=True,
            )
            return f"Error saving reference audio: {e}"
    elif tts_service == "xttsv2":
        logger.warning(
            f"XTTS-v2 selected for {name} ({Actor_id}) but no reference audio file uploaded."
        )
        return "Error: XTTS-v2 selected but no reference audio file uploaded."
    progress(0.7, desc="Saving character details to database...")
    await asyncio.to_thread(
        db_instance.save_character,
        name,
        personality,
        goals,
        backstory,
        tts_service,
        tts_model,
        reference_audio_filename,
        Actor_id,
        None,
    )
    token_msg_part = ""
    env_snippet = ""
    if Actor_id != "Actor1":
        token = await asyncio.to_thread(
            client_manager_instance.generate_token, Actor_id
        )
        if token:
            token_msg_part = f"Token for {Actor_id}: {token}"
            # Use selected IP if provided, else auto-pick
            server_url = "<your_server_url_here>"
            try:
                # Try to get the selected IP from the UI (Gradio passes it as an argument if wired)
                import inspect
                frame = inspect.currentframe()
                if frame is not None:
                    args, _, _, values = inspect.getargvalues(frame)
                    selected_ip = values.get('selected_server_ip', None)
                else:
                    selected_ip = None
            except Exception:
                selected_ip = None
            if selected_ip:
                server_url = f"http://{selected_ip}:8000"
            else:
                adapter_ips = Hardware.get_adapter_ip_addresses()
                if adapter_ips:
                    # Prefer Wi-Fi, Ethernet, or first available
                    preferred = None
                    for key in ["Wi-Fi", "Ethernet"]:
                        if key in adapter_ips:
                            preferred = adapter_ips[key]
                            break
                    if not preferred:
                        preferred = next(iter(adapter_ips.values()))
                    server_url = f"http://{preferred}:8000"
            env_snippet = f"CLIENT_Actor_ID=\"{Actor_id}\"\nCLIENT_TOKEN=\"{token}\"\nSERVER_URL=\"{server_url}\""
            logger.info(f"Generated token for {Actor_id}.")
    progress(1, desc="Character created!")
    logger.info(
        f"Character '{name}' for '{Actor_id}' created successfully. Token part: {token_msg_part}"
    )
    # Return both status and env_snippet for UI display
    return f"Character '{name}' for '{Actor_id}' created successfully. {token_msg_part}", env_snippet


async def story_interface_async(
    audio_input_path, chaos_level_value, progress=gr.Progress(track_tqdm=True)
):
    if csm_instance is None:
        raise RuntimeError("CSM instance not initialized.")
    if audio_input_path is None:
        return "No audio input. Record or upload narration.", {}, {}
    progress(0, desc="Transcribing narration audio...")
    try:
        # Always use the narrator's process_narration method for transcription
        if hasattr(csm_instance, "narrator") and hasattr(csm_instance.narrator, "process_narration"):
            narration_result = await csm_instance.narrator.process_narration(audio_input_path)
            transcription_text = narration_result.get("text", "")
        else:
            transcription_text = "[Narrator STT not available]"
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}", exc_info=True)
        transcription_text = f"[Transcription error: {e}]"
    progress(0.5, desc="Processing story turn...")
    try:
        narration, character_texts = await csm_instance.process_story(
            audio_input_path, chaos_level_value
        )
        progress(1, desc="Story turn processed.")
        if not isinstance(character_texts, dict):
            character_texts = {"error": "Invalid character text format from CSM"}
        logger.info(
            f"Story processed with audio: {audio_input_path}, chaos: {chaos_level_value}."
        )
        # Return transcription in the narrator's words textbox
        return transcription_text, character_texts
    except Exception as e:
        logger.error(
            f"Error in story_interface_async with audio {audio_input_path}: {e}",
            exc_info=True,
        )
        progress(1, desc="Error during story processing.")
        return f"Error: {e}", {}, {}


# ... other existing async handlers ...


async def save_correction_async(correction_text, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Saving correction...")
    if csm_instance is not None:
        logger.info(
            f"Attempting to save narration correction: {correction_text[:100]}..."
        )
        updated = await asyncio.to_thread(
            csm_instance.update_last_narration_text, correction_text
        )
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


async def refresh_story_async_wrapper(progress=gr.Progress()):
    progress(0, desc="Fetching history...")
    data = await get_story_playback_data_async()
    progress(1, desc="History loaded.")
    return data


async def save_checkpoint_async(name_prefix, progress=gr.Progress(track_tqdm=True)):
    if checkpoint_manager is None:
        logger.error(
            "CheckpointManager instance not initialized in save_checkpoint_async."
        )
        raise RuntimeError("CheckpointManager instance not initialized.")
    progress(0, desc="Saving checkpoint...")
    logger.info(f"Attempting to save checkpoint with prefix: {name_prefix}")
    status, new_choices = await asyncio.to_thread(
        checkpoint_manager.save_checkpoint, name_prefix
    )
    progress(1, desc="Checkpoint saved.")
    logger.info(f"Save checkpoint '{name_prefix}' result: {status}")
    return status, gr.update(
        choices=new_choices, value=new_choices[0] if new_choices else None
    )


async def load_checkpoint_async(checkpoint_name, progress=gr.Progress(track_tqdm=True)):
    if checkpoint_manager is None:
        logger.error(
            "CheckpointManager instance not initialized in load_checkpoint_async."
        )
        raise RuntimeError("CheckpointManager instance not initialized.")
    if not checkpoint_name:
        logger.warning("Load checkpoint attempted without selecting a checkpoint name.")
        return "Please select a checkpoint to load.", []
    progress(0, desc=f"Loading checkpoint '{checkpoint_name}'...")
    logger.info(f"Attempting to load checkpoint: {checkpoint_name}")
    status = await asyncio.to_thread(
        checkpoint_manager.load_checkpoint, checkpoint_name
    )
    new_story_data = []
    if status and "loaded" in status.lower() and "restart" not in status.lower():
        logger.info(
            f"Checkpoint '{checkpoint_name}' loaded successfully. Refreshing story history."
        )
        progress(0.8, desc="Refreshing story history...")
        new_story_data = await get_story_playback_data_async()
    else:
        logger.warning(
            f"Checkpoint '{checkpoint_name}' load status: {status}. Story history not refreshed."
        )
    progress(1, desc=f"Checkpoint '{checkpoint_name}' load attempt finished.")
    return status, new_story_data


async def export_story_async(export_format, progress=gr.Progress(track_tqdm=True)):
    if checkpoint_manager is None:
        logger.error(
            "CheckpointManager instance not initialized in export_story_async."
        )
        raise RuntimeError("CheckpointManager instance not initialized.")
    progress(0, desc=f"Exporting story as {export_format}...")
    logger.info(f"Attempting to export story as {export_format}.")
    status, filename = await asyncio.to_thread(
        checkpoint_manager.export_story, export_format
    )
    progress(1, desc="Story export finished.")
    logger.info(
        f"Export story as {export_format} result: {status}, filename: {filename}"
    )
    return status, gr.update(
        value=filename or "", visible=bool(filename)
    )  # Use gr.update


async def get_env_vars_async():
    status = await asyncio.to_thread(env_manager.get_env_file_status)
    masked_vars = await asyncio.to_thread(
        env_manager.load_env_vars, mask_sensitive=True
    )
    vars_display_str = (
        "\n".join([f"{k}={v}" for k, v in masked_vars.items()])
        or "# No variables found or .env file does not exist."
    )
    return status, vars_display_str


async def save_env_vars_async(new_vars_str: str, progress=gr.Progress()):
    progress(0, desc="Saving .env file...")
    logger.info("Attempting to save .env variables.")
    status_msg = await asyncio.to_thread(env_manager.save_env_vars, new_vars_str)
    progress(1, desc="Save complete.")
    logger.info(f".env file save result: {status_msg}")
    new_status, new_vars_display = await get_env_vars_async()
    return status_msg, new_status, new_vars_display


async def set_api_provider_async(selected_provider, progress=gr.Progress()):
    progress(0, desc="Updating API provider...")
    new_var = f"API_PROVIDER={selected_provider}"
    logger.info(f"Attempting to set API_PROVIDER to: {selected_provider}")
    status_msg = await asyncio.to_thread(env_manager.save_env_vars, new_var)
    progress(1, desc="Provider updated.")
    logger.info(f"API_PROVIDER update result: {status_msg}")
    new_status, new_vars_display = await get_env_vars_async()
    return status_msg, new_status, new_vars_display, selected_provider


async def restart_server_async(progress=gr.Progress()):
    progress(0, desc="Restarting server...")
    logger.info("Server restart requested via Gradio UI.")
    await asyncio.sleep(1)
    progress(1, desc="Server restarting now.")
    logger.info(f"Executing os.execv: {sys.executable} {sys.argv}")
    os.execv(sys.executable, [sys.executable] + sys.argv)
    return "Server is restarting..."


# --- Merged Character & Client Management Tab ---
def add_character_and_client_management_tab():
    """Adds the merged Character and Client Management tab to the Gradio interface."""
    with gr.TabItem("Character & Client Management"):
        gr.Markdown("## Manage Characters and Clients")
        with gr.Row():
            with gr.Column(scale=2):
                # Character Management Inputs
                char_name = gr.Textbox(label="Character Name", placeholder="E.g., 'Elara'")
                char_Actor_id = gr.Dropdown(
                    ["Actor1"] + [f"Actor{i}" for i in range(2, 11)],
                    label="Assign to Actor ID",
                    value="Actor1",
                )
                char_tts_service = gr.Dropdown(TTSManager.list_services(), label="TTS Service")
                char_tts_model = gr.Dropdown([], label="TTS Model")
                char_tts_voice = gr.Dropdown([], label="TTS Voice/Language")
                char_ref_audio = gr.File(
                    label="Reference Audio (XTTSv2)",
                    type="filepath",
                    visible=False,
                )
                # Adapter IP display and selection
                adapter_ip_btn = gr.Button("Show Adapter IPs")
                adapter_ip_output = gr.JSON(label="Adapter IP Addresses")
                adapter_ip_dropdown = gr.Dropdown(label="Select Server IP for Client .env", choices=[], visible=False)
                def update_adapter_ip_dropdown():
                    ips = Hardware.get_adapter_ip_addresses()
                    if not ips:
                        return gr.update(choices=[], visible=False), {}
                    return gr.update(choices=list(ips.values()), visible=True), ips
                adapter_ip_btn.click(update_adapter_ip_dropdown, inputs=[], outputs=[adapter_ip_dropdown, adapter_ip_output])
            with gr.Column(scale=3):
                char_personality = gr.Textbox(label="Personality", lines=2)
                char_goals = gr.Textbox(label="Goals", lines=2)
                char_backstory = gr.Textbox(label="Backstory", lines=3)
        create_char_btn = gr.Button("Save Character", variant="primary")
        char_creation_status = gr.Textbox(label="Status", interactive=False)
        char_env_snippet = gr.Textbox(label="Client .env Snippet (Copy & Paste)", lines=3, interactive=False)

        # Dynamic Client Management Inputs
        gr.Markdown("### Update Client Configurations via WebSocket")
        connected_clients_dropdown = gr.Dropdown(
            label="Select Client Actor_ID",
            choices=global_ws_connection_manager.get_active_clients(),
            allow_custom_value=True,
        )
        refresh_clients_btn = gr.Button("Refresh Client List")
        def update_client_list_ui():
            return gr.update(choices=global_ws_connection_manager.get_active_clients())
        refresh_clients_btn.click(update_client_list_ui, outputs=[connected_clients_dropdown])

        with gr.Row():
            with gr.Column():
                gr.Markdown("### TTS Configuration")
                new_tts_service = gr.Dropdown(
                    choices=[""] + TTSManager.list_services(),
                    label="New TTS Service (blank to skip)",
                )
                new_tts_model = gr.Textbox(label="New TTS Model Name (blank to skip)")
                new_tts_language = gr.Textbox(label="New TTS Language (e.g., 'en', 'es'; blank to skip)")
            with gr.Column():
                gr.Markdown("### Logging Configuration")
                new_log_level = gr.Dropdown(
                    choices=["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    label="New Log Level (blank to skip)",
                )
        send_config_update_btn = gr.Button("Send Configuration Update to Client", variant="primary")
        config_update_status = gr.Textbox(label="Update Status", interactive=False)

        # Handlers
        char_tts_service.change(
            fn=update_model_dropdown,
            inputs=char_tts_service,
            outputs=char_tts_model,
        )
        def update_voice_dropdown(service, model):
            voices = TTSManager.get_available_voices(service, model)
            default = voices[0] if voices else None
            return gr.update(choices=voices, value=default)
        char_tts_service.change(
            lambda service: gr.update(visible=(service == "xttsv2")),
            inputs=char_tts_service,
            outputs=char_ref_audio,
        )
        char_tts_service.change(
            lambda service: update_voice_dropdown(service, None),
            inputs=char_tts_service,
            outputs=char_tts_voice,
        )
        char_tts_model.change(
            lambda model, service: update_voice_dropdown(service, model),
            inputs=[char_tts_model, char_tts_service],
            outputs=char_tts_voice,
        )
        create_char_btn.click(
            create_character_async,
            inputs=[
                char_name,
                char_personality,
                char_goals,
                char_backstory,
                char_tts_service,
                char_tts_model,
                char_tts_voice,
                char_ref_audio,
                char_Actor_id,
                adapter_ip_dropdown,
            ],
            outputs=[char_creation_status, char_env_snippet],
        )
        async def handle_send_config_update(actor_id, tts_service, tts_model, tts_lang, log_level):
            if not actor_id:
                logger.warning("Dynamic config update: No Actor_ID selected/entered.")
                return "Error: No Actor_ID selected/entered."
            payload = {}
            if tts_service:
                payload["tts_service_name"] = tts_service
            if tts_model:
                payload["tts_model_name"] = tts_model
            if tts_lang:
                payload["tts_language"] = tts_lang
            if log_level:
                payload["log_level"] = log_level
            if not payload:
                logger.info("Dynamic config update: No configuration changes specified.")
                return "No configuration changes specified."
            message_to_send = {"type": "config_update", "payload": payload}
            logger.info(f"Attempting to send config update to {actor_id} via WebSocket: {message_to_send}")
            success = await global_ws_connection_manager.send_personal_message(message_to_send, actor_id)
            if success:
                logger.info(f"Configuration update WebSocket message sent to {actor_id}.")
                return f"Configuration update sent to {actor_id}."
            else:
                logger.warning(f"Failed to send configuration update WebSocket message to {actor_id}.")
                return f"Failed to send configuration update to {actor_id}. Client might be disconnected or an error occurred."
        send_config_update_btn.click(
            handle_send_config_update,
            inputs=[connected_clients_dropdown, new_tts_service, new_tts_model, new_tts_language, new_log_level],
            outputs=config_update_status,
        )


# --- Gradio UI Launch ---
def launch_interface():  # Renamed original launch_interface
    global db_instance, client_manager_instance, csm_instance, checkpoint_manager, env_manager_instance
    db_instance = Database(DB_PATH)
    client_manager_instance = ClientManager(db_instance)
    csm_instance = CSM()
    checkpoint_manager = CheckpointManager()
    env_manager_instance = env_manager

    import gradio.themes as themes

    with gr.Blocks(
        theme=themes.Soft(
            primary_hue=themes.colors.indigo, secondary_hue=themes.colors.blue
        )
    ) as demo:
        gr.Markdown("# Dream Weaver Interface")

        with gr.Tabs():
            add_character_and_client_management_tab()
            with gr.TabItem("Story Progression"):
                gr.Markdown("## Narrate the Story")
                with gr.Row():
                    with gr.Column(scale=2):
                        story_audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Record Narration",
                        )
                        story_chaos_slider = gr.Slider(
                            minimum=0, maximum=10, value=1, step=1, label="Chaos Level"
                        )
                        process_story_btn = gr.Button(
                            "Process Narration", variant="primary"
                        )
                    with gr.Column(scale=3):
                        narration_output_text = gr.Textbox(
                            label="Narrator's Words", lines=3, interactive=True
                        )
                        save_correction_btn = gr.Button(
                            "Save Correction", variant="secondary"
                        )
                        correction_status = gr.Textbox(
                            label="Correction Status", interactive=False
                        )
                        character_responses_json = gr.JSON(label="Character Dialogues")
                process_story_btn.click(
                    story_interface_async,
                    inputs=[story_audio_input, story_chaos_slider],
                    outputs=[narration_output_text, character_responses_json],
                )
                save_correction_btn.click(
                    save_correction_async,
                    inputs=[narration_output_text],
                    outputs=[correction_status],
                )

            with gr.TabItem("Story Playback & History"):
                gr.Markdown("## Review Story History")
                story_playback_chatbot = gr.Chatbot(
                    label="Full Story Log",
                    height=600,
                    show_copy_button=True,
                    type="messages",
                )
                refresh_story_btn = gr.Button("Refresh Story History")
                refresh_story_btn.click(
                    refresh_story_async_wrapper,
                    inputs=[],
                    outputs=[story_playback_chatbot],
                )
                demo.load(
                    refresh_story_async_wrapper,
                    inputs=[],
                    outputs=[story_playback_chatbot],
                )

            with gr.TabItem("System & Data Management"):
                gr.Markdown("## Checkpoints & Export")
                # ... (condensed UI as before)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Checkpoints")
                        chkpt_name_prefix = gr.Textbox(
                            label="Checkpoint Name Prefix",
                            placeholder="E.g., 'AfterTheHeist'",
                        )
                        save_chkpt_btn = gr.Button("Save Checkpoint")
                        current_checkpoints = checkpoint_manager.list_checkpoints()
                        chkpt_dropdown = gr.Dropdown(
                            choices=current_checkpoints,
                            label="Load Checkpoint",
                            value=(
                                current_checkpoints[0] if current_checkpoints else None
                            ),
                        )
                        load_chkpt_btn = gr.Button("Load Selected Checkpoint")
                        chkpt_status = gr.Textbox(
                            label="Checkpoint Status", interactive=False
                        )
                        save_chkpt_btn.click(
                            save_checkpoint_async,
                            inputs=[chkpt_name_prefix],
                            outputs=[chkpt_status, chkpt_dropdown],
                        )
                        load_chkpt_btn.click(
                            load_checkpoint_async,
                            inputs=[chkpt_dropdown],
                            outputs=[chkpt_status, story_playback_chatbot],
                        )
                    with gr.Column():
                        gr.Markdown("### Export Story")
                        export_format_radio = gr.Radio(
                            ["text", "json"], label="Export Format", value="text"
                        )
                        export_story_btn = gr.Button("Export Full Story")
                        export_status_text = gr.Textbox(label="Export Status")
                        export_filename_display = gr.Textbox(
                            label="Exported To"
                        )  # Changed to Textbox
                        export_story_btn.click(
                            export_story_async,
                            inputs=[export_format_radio],
                            outputs=[export_status_text, export_filename_display],
                        )

            with gr.TabItem("API Keys & .env"):
                gr.Markdown("## Manage Environment Variables (.env)")
                # ... (condensed UI as before)
                gr.Markdown(
                    "Add or update environment variables here... **Restart required...**"
                )
                env_status_text = gr.Textbox(
                    label=".env File Status", interactive=False
                )
                api_providers = ["huggingface", "openai", "google", "custom"]
                provider_token_vars = {
                    "huggingface": ("HUGGING_FACE_HUB_TOKEN", "HF Token", "hf_..."),
                    "openai": ("OPENAI_API_KEY", "OpenAI Key", "sk-..."),
                    "google": ("GOOGLE_API_KEY", "Google Key", "AIza..."),
                    "custom": ("CUSTOM_API_TOKEN", "Custom Token", "token..."),
                }

                def get_current_provider():
                    env_vars = env_manager.load_env_vars(mask_sensitive=False)
                    return env_vars.get("API_PROVIDER", api_providers[0])

                api_provider_dropdown = gr.Dropdown(
                    api_providers, label="API Provider", value=get_current_provider()
                )
                api_provider_status = gr.Textbox(
                    label="API Provider Status", interactive=False
                )
                token_input = gr.Textbox(
                    label="API Token", visible=True, placeholder="token..."
                )

                def update_token_field(provider):
                    var, label, placeholder = provider_token_vars.get(
                        provider, ("API_TOKEN", "API Token", "token...")
                    )
                    return gr.update(
                        visible=True, label=label, placeholder=placeholder, value=""
                    )

                api_provider_dropdown.change(
                    update_token_field,
                    inputs=[api_provider_dropdown],
                    outputs=[token_input],
                )

                async def save_token_async(provider, token, progress=gr.Progress()):
                    progress(0, desc="Saving token...")
                    var, _, _ = provider_token_vars.get(
                        provider, ("API_TOKEN", "API Token", "token...")
                    )
                    new_var = f"{var}={token}"
                    logger.info(
                        f"Attempting to save token for provider: {provider} (variable: {var})"
                    )
                    status_msg = await asyncio.to_thread(
                        env_manager.save_env_vars, new_var
                    )
                    progress(1, desc="Token saved.")
                    logger.info(f"Save token for {provider} result: {status_msg}")
                    new_status, new_vars_display = await get_env_vars_async()
                    return status_msg, new_status, new_vars_display, ""

                save_token_btn = gr.Button("Save API Token", variant="primary")
                save_token_status = gr.Textbox(
                    label="Token Save Status", interactive=False
                )
                with gr.Row():
                    with gr.Column():
                        current_env_vars_display = gr.Textbox(
                            label="Current .env Content",
                            lines=10,
                            interactive=False,
                            placeholder="# .env content...",
                        )
                    with gr.Column():
                        new_env_vars_input = gr.Textbox(
                            label="New/Updated Variables",
                            lines=10,
                            placeholder="KEY=value...",
                        )
                save_env_btn = gr.Button("Save to .env", variant="primary")
                save_status_text = gr.Textbox(label="Save Status", interactive=False)
                restart_required_text = gr.Markdown(
                    "**Restart required...**", visible=False
                )
                restart_btn = gr.Button("Restart Server", variant="stop", visible=False)

                def show_restart():
                    return gr.update(visible=True), gr.update(visible=True)

                demo.load(
                    get_env_vars_async,
                    inputs=[],
                    outputs=[env_status_text, current_env_vars_display],
                )
                save_env_btn.click(
                    save_env_vars_async,
                    inputs=[new_env_vars_input],
                    outputs=[
                        save_status_text,
                        env_status_text,
                        current_env_vars_display,
                    ],
                ).then(
                    show_restart,
                    inputs=[],
                    outputs=[restart_required_text, restart_btn],
                )
                api_provider_dropdown.change(
                    set_api_provider_async,
                    inputs=[api_provider_dropdown],
                    outputs=[
                        api_provider_status,
                        env_status_text,
                        current_env_vars_display,
                        api_provider_dropdown,
                    ],
                )
                save_token_btn.click(
                    save_token_async,
                    inputs=[api_provider_dropdown, token_input],
                    outputs=[
                        save_token_status,
                        env_status_text,
                        current_env_vars_display,
                        token_input,
                    ],
                ).then(
                    show_restart,
                    inputs=[],
                    outputs=[restart_required_text, restart_btn],
                )
                restart_btn.click(restart_server_async, inputs=[], outputs=[])

            with gr.TabItem("Config & Model Options"):
                gr.Markdown("## Edit Config & Model Options")
                # ... (condensed UI as before)
                from .config import EDITABLE_CONFIG_OPTIONS

                config_keys = list(EDITABLE_CONFIG_OPTIONS.keys())
                config_values = [str(EDITABLE_CONFIG_OPTIONS[k]) for k in config_keys]
                config_inputs = [
                    gr.Textbox(label=k, value=v)
                    for k, v in zip(config_keys, config_values)
                ]
                save_config_btn = gr.Button("Save Config Changes", variant="primary")
                config_status = gr.Textbox(
                    label="Config Save Status", interactive=False
                )

                async def save_config_async(*new_values):
                    lines = []
                    for k, v in zip(config_keys, new_values):
                        lines.append(f"{k}={v}")
                    logger.info(f"Attempting to save config changes to .env: {lines}")
                    status_msg = await asyncio.to_thread(
                        env_manager.save_env_vars, "\n".join(lines)
                    )
                    logger.info(f"Save config changes result: {status_msg}")
                    return status_msg

                save_config_btn.click(
                    save_config_async, inputs=config_inputs, outputs=[config_status]
                )

        gr.Markdown("---")
        gr.Markdown("View [Server Dashboard](/dashboard) (Server Perf & Client Status)")

    demo.queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    logger.info(
        "Launching Gradio interface directly (SERVER/src/gradio_interface.py)..."
    )
    launch_interface()  # Call the original launch_interface directly
