import gradio as gr
from .csm import CSM
from .database import Database
from .tts_manager import TTSManager # Server's TTSManager
from .client_manager import ClientManager
from .checkpoint_manager import CheckpointManager
from .config import DB_PATH, REFERENCE_VOICES_AUDIO_PATH
import shutil
import os
import asyncio # Added asyncio

# --- Instances ---
# These are global instances; care must be taken if server scales to multiple workers.
# For typical Gradio/FastAPI on Uvicorn with multiple workers, these might need
# to be managed differently (e.g., per-request or using a shared service pattern).
# However, for a single-worker setup or simple multi-threading, this can work.
db_instance = Database(DB_PATH)
client_manager_instance = ClientManager(db_instance) # ClientManager needs db
csm_instance = CSM() # CSM initializes its own db, narrator, character_server, client_manager
checkpoint_manager = CheckpointManager() # CheckpointManager uses paths from server config

# --- Helper Functions (mostly synchronous as they are simple UI updates or fast DB calls) ---
def update_model_dropdown(service_name: str):
    """Dynamically updates the TTS model dropdown based on the selected service."""
    # Uses Server's TTSManager to list models
    models = TTSManager.get_available_models(service_name)
    default_value = models[0] if models else None
    return gr.Dropdown.update(choices=models, value=default_value)

def get_story_playback_data_sync(): # Renamed to indicate it's synchronous
    """Fetches story history from DB and formats it for Gradio Chatbot."""
    # This is a DB read, usually fast. If it becomes slow for very long stories,
    # it could be made async with to_thread and a progress indicator.
    history_raw = db_instance.get_story_history()
    chatbot_messages = []
    if not history_raw:
        return [["System", "No story history found."]]

    for entry in history_raw:
        speaker = entry.get("speaker", "Unknown")
        text = entry.get("text", "")
        timestamp = entry.get("timestamp", "")
        # Format for chatbot
        formatted_text = f"_{timestamp}_ \n**{speaker}:** {text}"
        if speaker.lower() == "narrator":
            chatbot_messages.append([formatted_text, None])
        else:
            if chatbot_messages and chatbot_messages[-1][0] is not None and chatbot_messages[-1][1] is None:
                 chatbot_messages[-1][1] = formatted_text
            else:
                chatbot_messages.append([None, formatted_text])
    return chatbot_messages

# --- Asynchronous Gradio Event Handlers ---

async def create_character_async(name, personality, goals, backstory, tts_service, tts_model, reference_audio_file, pc_id, progress=gr.Progress(track_tqdm=True)):
    """Asynchronously creates a character, handles file copy with progress."""
    progress(0, desc="Initializing character creation...")

    reference_audio_filename = None
    if reference_audio_file and tts_service == "xttsv2":
        progress(0.2, desc="Processing reference audio...")
        # Ensure the directory from config exists (config.py does this, but good practice)
        os.makedirs(REFERENCE_VOICES_AUDIO_PATH, exist_ok=True)
        sane_name = "".join(c if c.isalnum() else "_" for c in name)
        original_filename = reference_audio_file.name
        _, ext = os.path.splitext(original_filename)
        reference_audio_filename = f"{sane_name}_{pc_id}_{os.urandom(4).hex()}{ext}"
        destination_path = os.path.join(REFERENCE_VOICES_AUDIO_PATH, reference_audio_filename)

        try:
            # shutil.copyfile is blocking, run in thread
            await asyncio.to_thread(shutil.copyfile, original_filename, destination_path)
            print(f"Saved reference audio to: {destination_path}")
            progress(0.5, desc="Reference audio saved.")
        except Exception as e:
            print(f"Error saving reference audio: {e}")
            return f"Error saving reference audio: {e}"
    elif tts_service == "xttsv2" and not reference_audio_file:
        return "Error: XTTS-v2 selected but no reference audio file uploaded."

    progress(0.7, desc="Saving character details to database...")
    # Database operations are fast, but can be threaded for consistency if desired
    await asyncio.to_thread(
        db_instance.save_character,
        name, personality, goals, backstory, tts_service, tts_model,
        reference_audio_filename, pc_id,
        # Assuming a default or None for llm_model if not provided by UI yet
        # You might want to add an llm_model dropdown in the UI too
        character.get("llm_model") if 'character' in locals() and character else None
    )

    token_msg_part = ""
    if pc_id != "PC1":
        # Token generation is fast (secrets.token_hex + DB write)
        token = await asyncio.to_thread(client_manager_instance.generate_token, pc_id)
        if token:
            token_msg_part = f" Token for {pc_id}: {token}"

    progress(1, desc="Character created!")
    return f"Character '{name}' for '{pc_id}' created successfully.{token_msg_part}"


async def story_interface_async(audio_input_path, chaos_level_value, progress=gr.Progress(track_tqdm=True)):
    """Asynchronously processes the story turn with progress updates."""
    if audio_input_path is None:
        return "No audio input. Record or upload narration.", {}, gr.Button.update() # No change to button

    progress(0, desc="Starting story processing...")
    # CSM's process_story is now async
    try:
        # Simulate progress for different stages if csm.process_story doesn't have internal progress reporting
        # For a real gr.Progress with tqdm, csm.process_story would need to accept a tqdm instance
        # or use internal tqdm compatible logging.
        # For now, we'll just await the whole thing.
        # To show more granular progress, you'd break down csm.process_story or have it yield updates.

        # Placeholder for narrator progress (assuming it's part of process_story)
        # progress(0.1, desc="Transcribing narration...")

        narration, character_texts = await csm_instance.process_story(audio_input_path, chaos_level_value)

        progress(1, desc="Story turn processed.")

        if not isinstance(character_texts, dict): # Should not happen if csm is correct
            character_texts = {"error": "Invalid character text format from CSM"}

        return narration, character_texts
    except Exception as e:
        print(f"Error in story_interface_async: {e}")
        progress(1, desc="Error during story processing.") # Clear progress
        return f"Error: {e}", {}, gr.Button.update() # Update button state if needed


async def save_checkpoint_async(name_prefix, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Saving checkpoint...")
    # CheckpointManager.save_checkpoint involves file I/O (blocking)
    status, new_choices = await asyncio.to_thread(checkpoint_manager.save_checkpoint, name_prefix)
    progress(1, desc="Checkpoint saved.")
    return status, gr.Dropdown.update(choices=new_choices, value=new_choices[0] if new_choices else None)

async def load_checkpoint_async(checkpoint_name, progress=gr.Progress(track_tqdm=True)):
    if not checkpoint_name:
        return "Please select a checkpoint to load.", [] # Return empty list for chatbot if no load
    progress(0, desc=f"Loading checkpoint '{checkpoint_name}'...")
    # CheckpointManager.load_checkpoint involves file I/O (blocking)
    status = await asyncio.to_thread(checkpoint_manager.load_checkpoint, checkpoint_name)

    new_story_data = []
    if "loaded" in status.lower() and "restart" not in status.lower() : # If successfully loaded and no restart needed (or handle restart message)
        progress(0.8, desc="Refreshing story history...")
        # get_story_playback_data_sync is blocking (DB read)
        new_story_data = await asyncio.to_thread(get_story_playback_data_sync)

    progress(1, desc=f"Checkpoint '{checkpoint_name}' load attempt finished.")
    return status, new_story_data

async def export_story_async(export_format, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc=f"Exporting story as {export_format}...")
    # CheckpointManager.export_story involves DB read and file write (blocking)
    status, filename = await asyncio.to_thread(checkpoint_manager.export_story, export_format)
    progress(1, desc="Story export finished.")
    return status, gr.Textbox.update(value=filename if filename else "", visible=bool(filename))


# --- Gradio UI Launch ---
def launch_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.indigo, secondary_hue=gr.themes.colors.blue)) as demo:
        gr.Markdown("# Dream Weaver Interface")

        with gr.Tabs():
            with gr.TabItem("Character Management"):
                # ... (Character Management UI definition - use create_character_async) ...
                gr.Markdown("## Create or Update Characters")
                with gr.Row():
                    with gr.Column(scale=2):
                        char_name = gr.Textbox(label="Character Name", placeholder="E.g., 'Elara'")
                        char_pc_id = gr.Dropdown(["PC1"] + [f"PC{i}" for i in range(2, 11)], label="Assign to PC ID", value="PC1")
                        char_tts_service = gr.Dropdown(TTSManager.list_services(), label="TTS Service")
                        char_tts_model = gr.Dropdown([], label="TTS Model", interactive=True)
                        char_ref_audio = gr.File(label="Reference Audio (XTTSv2)", type="filepath", interactive=True, visible=False)
                    with gr.Column(scale=3):
                        char_personality = gr.Textbox(label="Personality", lines=2)
                        char_goals = gr.Textbox(label="Goals", lines=2)
                        char_backstory = gr.Textbox(label="Backstory", lines=3)

                create_char_btn = gr.Button("Save Character", variant="primary")
                char_creation_status = gr.Textbox(label="Status", interactive=False)

                char_tts_service.change(fn=update_model_dropdown, inputs=char_tts_service, outputs=char_tts_model)
                char_tts_service.change(lambda service: gr.File.update(visible=(service == "xttsv2")), inputs=char_tts_service, outputs=char_ref_audio)

                create_char_btn.click(
                    create_character_async,
                    inputs=[char_name, char_personality, char_goals, char_backstory, char_tts_service, char_tts_model, char_ref_audio, char_pc_id],
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
                        narration_output_text = gr.Textbox(label="Narrator's Words", lines=3, interactive=False)
                        character_responses_json = gr.JSON(label="Character Dialogues", interactive=False) # Changed from Textbox

                process_story_btn.click(
                    story_interface_async,
                    inputs=[story_audio_input, story_chaos_slider],
                    outputs=[narration_output_text, character_responses_json]
                )

            with gr.TabItem("Story Playback & History"):
                # ... (Story Playback UI - get_story_playback_data_sync is likely fine for initial load and refresh unless very slow) ...
                gr.Markdown("## Review Story History")
                story_playback_chatbot = gr.Chatbot(label="Full Story Log", height=600, show_copy_button=True, bubble_full_width=False)
                refresh_story_btn = gr.Button("Refresh Story History")

                # Making this async too for consistency, though DB reads are usually fast.
                async def refresh_story_async_wrapper(progress=gr.Progress()):
                    progress(0, desc="Fetching history...")
                    data = await asyncio.to_thread(get_story_playback_data_sync)
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
                        export_status_text = gr.Textbox(label="Export Status", interactive=False)
                        export_filename_display = gr.Textbox(label="Exported To", interactive=False)

                        export_story_btn.click(
                            export_story_async,
                            inputs=[export_format_radio],
                            outputs=[export_status_text, export_filename_display]
                        )

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
