import gradio as gr
from .csm import CSM
from .database import Database
from .tts_manager import TTSManager
from .client_manager import ClientManager
from .checkpoint_manager import CheckpointManager
from .config import DB_PATH, REFERENCE_VOICES_AUDIO_PATH # Import from config
import shutil
import os

# Use DB_PATH from config
db_instance = Database(DB_PATH)
client_manager_instance = ClientManager(db_instance)
# CSM now uses DB_PATH from config internally
csm_instance = CSM()
# CheckpointManager now uses paths from config internally
checkpoint_manager = CheckpointManager()

def update_model_dropdown(service):
    """Dynamically updates the model dropdown based on the selected TTS service."""
    if service == "piper":
        models = TTSManager.get_available_models(service)
    elif service == "xttsv2":
        models = TTSManager.get_available_models(service) # Should be ["tts_models/multilingual/multi-dataset/xtts_v2"]
    elif service == "google":
        models = TTSManager.get_available_models(service) # e.g., ['en-US-Standard-A', ...]
    else:
        models = []

    # Ensure a valid default is selected if models list is not empty
    default_value = models[0] if models else None
    return gr.Dropdown.update(choices=models, value=default_value)


def create_character(name, personality, goals, backstory, tts, tts_model, reference_audio_file, pc):
    reference_audio_filename = None
    if reference_audio_file and tts == "xttsv2": # Only save if XTTSv2 is selected and file is provided
        # Ensure the directory from config exists
        os.makedirs(REFERENCE_VOICES_AUDIO_PATH, exist_ok=True)

        # Sanitize character name for use in filename
        sane_name = "".join(c if c.isalnum() else "_" for c in name)

        # Get original extension
        original_filename = reference_audio_file.name # This is the temporary path Gradio provides
        _, ext = os.path.splitext(original_filename)

        # Create a more robust unique filename
        reference_audio_filename = f"{sane_name}_{os.urandom(4).hex()}{ext}"
        destination_path = os.path.join(REFERENCE_VOICES_AUDIO_PATH, reference_audio_filename)

        try:
            shutil.copyfile(original_filename, destination_path) # Gradio provides a temp file path
            print(f"Saved reference audio to: {destination_path}")
        except Exception as e:
            print(f"Error saving reference audio: {e}")
            return f"Error saving reference audio: {e}"

    elif tts == "xttsv2" and not reference_audio_file:
        return "Error: XTTS-v2 selected but no reference audio file uploaded. Please upload a voice sample."


    db_instance.save_character(name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc)
    token = None
    if pc != "PC1": # Assuming "PC1" is the server/local character
        token = client_manager_instance.generate_token(pc)

    msg = f"Character '{name}' for '{pc}' created successfully."
    if token:
        msg += f" Token for {pc}: {token}"
    return msg


def story_interface(audio_input_path, chaos_level_value):
    if audio_input_path is None:
        return "No audio input provided. Please record or upload narration.", {}

    # CSM's process_story expects the audio filepath and chaos level
    narration, character_texts = csm_instance.process_story(audio_input_path, chaos_level_value)

    # Ensure character_texts is a dictionary (it should be)
    if not isinstance(character_texts, dict):
        character_texts = {"error": "Invalid character text format from CSM"}

    return narration, character_texts


def get_story_playback_data():
    """Fetches story history from DB and formats it for Gradio Chatbot."""
    history_raw = db_instance.get_story_history() # Ensure this method exists and works
    chatbot_messages = []
    if not history_raw:
        return [["System", "No story history found."]]

    for entry in history_raw:
        # Assuming entry structure: (speaker, text, timestamp, audio_path_or_none)
        speaker, text, timestamp = entry[0], entry[1], entry[2]

        # Format for chatbot: [user_message, bot_response]
        # Narrator's text appears on the left (user side), character's on the right (bot side).
        formatted_text = f"_{timestamp}_ \n**{speaker}:** {text}"
        if speaker.lower() == "narrator": # Case-insensitive check for narrator
            chatbot_messages.append([formatted_text, None])
        else:
            # If there's a previous narrator message without a response, append to it
            if chatbot_messages and chatbot_messages[-1][0] is not None and chatbot_messages[-1][1] is None:
                 chatbot_messages[-1][1] = formatted_text
            else: # Otherwise, start a new line with None for user part (or handle as per desired UI)
                chatbot_messages.append([None, formatted_text])
    return chatbot_messages


def save_checkpoint_handler(name_prefix):
    status, new_choices = checkpoint_manager.save_checkpoint(name_prefix)
    return status, gr.Dropdown.update(choices=new_choices, value=new_choices[0] if new_choices else None)

def load_checkpoint_handler(checkpoint_name):
    if not checkpoint_name:
        return "Please select a checkpoint to load."
    status = checkpoint_manager.load_checkpoint(checkpoint_name)
    # After loading, refresh story playback as DB might have changed
    new_story_data = get_story_playback_data()
    return status, new_story_data


def launch_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Dream Weaver Interface")

        with gr.Tabs():
            with gr.TabItem("Character Management"):
                gr.Markdown("## Create or Update Characters")
                with gr.Row():
                    with gr.Column(scale=2):
                        char_name = gr.Textbox(label="Character Name", placeholder="E.g., 'Elara the Explorer'")
                        char_pc = gr.Dropdown(["PC1"] + [f"PC{i}" for i in range(2, 11)], label="Assign to PC", value="PC1", info="PC1 is the server. Others are clients.")
                        char_tts_service = gr.Dropdown(TTSManager.list_services(), label="TTS Service", info="Select the Text-to-Speech engine.")
                        char_tts_model = gr.Dropdown([], label="TTS Model", interactive=True, info="Model for the selected TTS service.")
                        char_ref_audio = gr.File(label="Reference Audio (for XTTS-v2 voice cloning)", type="filepath", interactive=True, visible=False)
                    with gr.Column(scale=3):
                        char_personality = gr.Textbox(label="Personality Traits", placeholder="E.g., 'Brave, curious, slightly reckless'", lines=2)
                        char_goals = gr.Textbox(label="Character Goals", placeholder="E.g., 'Find the lost temple, protect her crew'", lines=2)
                        char_backstory = gr.Textbox(label="Brief Backstory", placeholder="E.g., 'Left her village after discovering an ancient map...'", lines=3)

                create_char_btn = gr.Button("Save Character Configuration", variant="primary")
                char_creation_status = gr.Textbox(label="Status", interactive=False)

                # Dynamic updates for TTS model dropdown and reference audio visibility
                char_tts_service.change(fn=update_model_dropdown, inputs=char_tts_service, outputs=char_tts_model)
                char_tts_service.change(lambda service: gr.File.update(visible=(service == "xttsv2")), inputs=char_tts_service, outputs=char_ref_audio)

                create_char_btn.click(
                    create_character,
                    inputs=[char_name, char_personality, char_goals, char_backstory, char_tts_service, char_tts_model, char_ref_audio, char_pc],
                    outputs=char_creation_status
                )
                # TODO: Add a way to view/edit existing characters

            with gr.TabItem("Story Progression"):
                gr.Markdown("## Narrate the Story")
                with gr.Row():
                    with gr.Column(scale=2):
                        story_audio_input = gr.Audio(source="microphone", type="filepath", label="Record Narration", show_label=True)
                        story_chaos_slider = gr.Slider(minimum=0, maximum=10, value=1, step=1, label="Chaos Level", info="Higher level increases chances of random events.")
                        process_story_btn = gr.Button("Process Narration", variant="primary")
                    with gr.Column(scale=3):
                        narration_output_text = gr.Textbox(label="Narrator's Words", lines=3, interactive=False)
                        character_responses_json = gr.JSON(label="Character Dialogues", interactive=False)

                process_story_btn.click(
                    story_interface,
                    inputs=[story_audio_input, story_chaos_slider],
                    outputs=[narration_output_text, character_responses_json]
                )

            with gr.TabItem("Story Playback & History"):
                gr.Markdown("## Review Story History")
                story_playback_chatbot = gr.Chatbot(label="Full Story Log", height=500, show_copy_button=True, bubble_full_width=False)
                refresh_story_btn = gr.Button("Refresh Story History")

                refresh_story_btn.click(get_story_playback_data, inputs=[], outputs=[story_playback_chatbot])
                # Load history once when the interface loads
                demo.load(get_story_playback_data, inputs=[], outputs=[story_playback_chatbot])


            with gr.TabItem("System & Data Management"):
                gr.Markdown("## Checkpoints & Export")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Checkpoints")
                        chkpt_name_prefix = gr.Textbox(label="Checkpoint Name Prefix", placeholder="E.g., 'AfterTheCave'")
                        save_chkpt_btn = gr.Button("Save Current State as Checkpoint")

                        active_checkpoints = checkpoint_manager.list_checkpoints()
                        chkpt_dropdown = gr.Dropdown(choices=active_checkpoints, label="Load from Checkpoint", value=active_checkpoints[0] if active_checkpoints else None)
                        load_chkpt_btn = gr.Button("Load Selected Checkpoint")
                        chkpt_status = gr.Textbox(label="Checkpoint Status", interactive=False)

                        save_chkpt_btn.click(
                            save_checkpoint_handler,
                            inputs=[chkpt_name_prefix],
                            outputs=[chkpt_status, chkpt_dropdown]
                        )
                        # When loading a checkpoint, also refresh the story playback
                        load_chkpt_btn.click(
                            load_checkpoint_handler,
                            inputs=[chkpt_dropdown],
                            outputs=[chkpt_status, story_playback_chatbot] # Update status and chatbot
                        )

                    with gr.Column():
                        gr.Markdown("### Export Story")
                        export_format_radio = gr.Radio(["text", "json"], label="Choose Export Format", value="text")
                        export_story_btn = gr.Button("Export Full Story")
                        export_status_text = gr.Textbox(label="Export Status", interactive=False)
                        # Keep filename output hidden as direct download isn't standard. Status gives filename.
                        export_filename_display = gr.Textbox(label="Exported To", interactive=False, visible=True)

                        def export_story_wrapper(fmt):
                            status, filename = checkpoint_manager.export_story(fmt)
                            # Make filename visible only if export was successful
                            return status, gr.Textbox.update(value=filename if filename else "", visible=bool(filename))

                        export_story_btn.click(
                            export_story_wrapper,
                            inputs=[export_format_radio],
                            outputs=[export_status_text, export_filename_display]
                        )

        gr.Markdown("---")
        gr.Markdown("View [Monitoring Dashboard](/dashboard) (Server & Client Status)")


    demo.launch(server_name="0.0.0.0") # Make it accessible on the network

if __name__ == "__main__":
    # This allows running the Gradio interface directly for testing if needed
    # Ensure the necessary paths (like for DB) are resolvable if run this way
    # or that environment variables are set.
    print("Launching Gradio interface directly...")
    # Make sure config creates directories before DB is initialized
    from .config import DB_PATH, REFERENCE_VOICES_AUDIO_PATH, CHARACTERS_AUDIO_PATH, MODELS_PATH, ADAPTERS_PATH, BASE_CHECKPOINT_PATH, NARRATOR_AUDIO_PATH
    # The config.py already runs os.makedirs, so just importing it should be enough.

    launch_interface()
