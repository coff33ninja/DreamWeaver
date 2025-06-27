import gradio as gr
from .csm import CSM
from .database import Database
from .tts_manager import TTSManager
from .client_manager import ClientManager
from .checkpoint_manager import CheckpointManager
import shutil
import os

db_instance = Database("E:/DreamWeaver/data/dream_weaver.db")
client_manager_instance = ClientManager(db_instance)
csm_instance = CSM()
checkpoint_manager = CheckpointManager()

def update_model_dropdown(service):
    """Dynamically updates the model dropdown based on the selected TTS service."""
    if service == "piper":
        # A curated list of good Piper voices
        models = ["en_US-ryan-high", "en_US-lessac-medium", "en_US-joe-medium", "en_GB-alan-low"]
    elif service == "xttsv2":
        # XTTSv2 has one primary model
        models = ["tts_models/multilingual/multi-dataset/xtts_v2"]
    else:
        models = []
    return gr.Dropdown.update(choices=models, value=models[0] if models else None)

def create_character(name, personality, goals, backstory, tts, tts_model, reference_audio_file, pc, db=db_instance, client_manager=client_manager_instance):
    reference_audio_filename = None
    if reference_audio_file:
        # Save the uploaded audio file to a designated directory
        ref_audio_dir = "E:/DreamWeaver/data/audio/reference_voices"
        os.makedirs(ref_audio_dir, exist_ok=True)

        # Use a unique filename to avoid collisions
        base, ext = os.path.splitext(reference_audio_file.name)
        reference_audio_filename = f"{name}_{base}_{os.urandom(4).hex()}{ext}"
        destination_path = os.path.join(ref_audio_dir, reference_audio_filename)
        shutil.copyfile(reference_audio_file.name, destination_path)
        print(f"Saved reference audio to: {destination_path}")

    db.save_character(name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc)
    token = None
    if pc != "PC1":
        token = client_manager.generate_token(pc)
    return f"Created {name} on {pc}" + (f" | Token: {token}" if token else "")

def story_interface(audio, chaos_level, csm=csm_instance):
    narration, character_texts = csm.process_story(audio, chaos_level)
    return narration, {k: v for k, v in character_texts.items()}

def get_story_playback_data():
    """Fetches story history from DB and formats it for Gradio Chatbot."""
    history_raw = db_instance.get_story_history()
    chatbot_messages = []
    for entry in history_raw:
        speaker, text, timestamp = entry
        # Format for chatbot: [user_message, bot_response]
        # Narrator's text appears on the left (user side), character's on the right (bot side).
        formatted_text = f"[{timestamp}] {speaker}: {text}"
        if speaker == "Narrator":
            chatbot_messages.append([formatted_text, None])
        else:
            chatbot_messages.append([None, formatted_text])
    return chatbot_messages

def save_checkpoint_handler(name_prefix):
    status, new_choices = checkpoint_manager.save_checkpoint(name_prefix)
    return status, gr.Dropdown.update(choices=new_choices)

def load_checkpoint_handler(checkpoint_name):
    if not checkpoint_name:
        return "Please select a checkpoint to load."
    return checkpoint_manager.load_checkpoint(checkpoint_name)

def launch_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Dream Weaver: Character Creation")
        with gr.Row():
            name = gr.Textbox(label="Character Name")
            personality = gr.Textbox(label="Personality")
            goals = gr.Textbox(label="Goals")
            backstory = gr.Textbox(label="Backstory")
            tts_service_dd = gr.Dropdown(TTSManager.list_services(), label="TTS Service")
            tts_model_dd = gr.Dropdown([], label="TTS Model")
            reference_audio_upload = gr.File(label="Upload Reference Audio (for XTTS-v2 cloning)", type="filepath", visible=False)
            pc = gr.Dropdown(["PC1"] + [f"PC{i}" for i in range(2, 10)], label="PC")
            create_btn = gr.Button("Create Character")

        tts_service_dd.change(fn=update_model_dropdown, inputs=tts_service_dd, outputs=tts_model_dd)
        tts_service_dd.change(lambda service: gr.File.update(visible=(service == "xttsv2")), inputs=tts_service_dd, outputs=reference_audio_upload)
        create_btn.click(create_character, inputs=[name, personality, goals, backstory, tts_service_dd, tts_model_dd, reference_audio_upload, pc], outputs=gr.Textbox())

        gr.Markdown("## Dream Weaver: Story")
        audio_input = gr.Audio(source="microphone", label="Narrate")
        chaos_level = gr.Slider(0, 10, label="Chaos Level")
        narration_output = gr.Textbox(label="Narration")
        character_outputs = gr.JSON(label="Character Dialogues")
        story_btn = gr.Button("Process Story")
        story_btn.click(story_interface, inputs=[audio_input, chaos_level], outputs=[narration_output, character_outputs])

        gr.Markdown("## Dream Weaver: Checkpoints")
        with gr.Row():
            checkpoint_name_prefix = gr.Textbox(label="Checkpoint Name (Prefix)", placeholder="e.g., AfterTheHeist")
            save_checkpoint_btn = gr.Button("Save Checkpoint")
        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(choices=checkpoint_manager.list_checkpoints(), label="Load Checkpoint")
            load_checkpoint_btn = gr.Button("Load Checkpoint")
        checkpoint_status = gr.Textbox(label="Status", interactive=False)

        save_checkpoint_btn.click(save_checkpoint_handler, inputs=[checkpoint_name_prefix], outputs=[checkpoint_status, checkpoint_dropdown])
        load_checkpoint_btn.click(load_checkpoint_handler, inputs=[checkpoint_dropdown], outputs=[checkpoint_status])

        gr.Markdown("## Dream Weaver: Story Playback")
        with gr.Row():
            story_chatbot = gr.Chatbot(label="Story History", height=400, show_copy_button=True)
        with gr.Row():
            refresh_story_btn = gr.Button("Refresh Story History")

        refresh_story_btn.click(get_story_playback_data, inputs=[], outputs=[story_chatbot])
        demo.load(get_story_playback_data, inputs=[], outputs=[story_chatbot]) # Initial load on startup

        gr.Markdown("## Dream Weaver: Export Story")
        with gr.Row():
            export_format_radio = gr.Radio(["text", "json"], label="Export Format", value="text")
            export_story_btn = gr.Button("Export Story")
        export_status = gr.Textbox(label="Export Status", interactive=False)
        export_filename_output = gr.Textbox(label="Exported Filename", interactive=False, visible=False)  # Hidden initially

        def export_story_handler(export_format):
            status, filename = checkpoint_manager.export_story(export_format)
            return status, gr.Textbox.update(value=filename, visible=(filename is not None))

        export_story_btn.click(export_story_handler, inputs=[export_format_radio], outputs=[export_status, export_filename_output])


    demo.launch()
