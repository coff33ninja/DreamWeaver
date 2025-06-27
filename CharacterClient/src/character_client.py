from fastapi import FastAPI, HTTPException
import requests
import uuid
import os
import sys
import base64
import asyncio # For background tasks
import tempfile # For temporary file storage

# Import client-specific modules
from tts_manager import TTSManager
from llm_engine import LLMEngine

app = FastAPI()
SERVER_URL = os.getenv("SERVER_URL", "http://192.168.1.101:8000")
CLIENT_PC_ID = os.getenv("CLIENT_PC_ID", "PC2") # Default, should be set per client

HEARTBEAT_INTERVAL_SECONDS = 60 # Send a heartbeat every 60 seconds

class CharacterClient:

    def fetch_traits(self):
        response = requests.get(f"{SERVER_URL}/get_traits", params={"pc": self.pc_id, "token": self.token})
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=401, detail="Invalid token")

    def _register_with_server(self):
        """Announces the client's presence to the server."""
        print(f"Sending heartbeat from {self.pc_id}...")
        try:
            response = requests.post(f"{SERVER_URL}/register", json={"pc": self.pc_id, "token": self.token})
            response.raise_for_status()
            print(response.json()["message"])
            return True
        except requests.exceptions.RequestException as e:
            print(f"Could not register with server: {e}")
            return False

    def __init__(self, token: str):
        self.token = token
        self.pc_id = CLIENT_PC_ID # Use the configured PC ID
        self._register_with_server()
        self.local_reference_audio_path = None # Store path to downloaded reference audio
        self.character = self.fetch_traits()
        self.tts = TTSManager(self.character.get("tts"), self.character.get("tts_model"))
        self.llm = LLMEngine()

        # If XTTS-v2 and reference audio is specified, download it
        if self.character.get("tts") == "xttsv2" and self.character.get("reference_audio_filename"):
            try:
                ref_audio_url = f"{SERVER_URL}/get_reference_audio/{self.character['reference_audio_filename']}"
                print(f"Downloading reference audio from: {ref_audio_url}")
                response = requests.get(ref_audio_url, stream=True)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                    self.local_reference_audio_path = tmp_file.name
                print(f"Reference audio downloaded to: {self.local_reference_audio_path}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading reference audio: {e}")
                self.local_reference_audio_path = None

    def generate_response(self, narration, character_texts):
        prompt = f"Narrator: {narration}\n" + "\n".join([f"{k}: {v}" for k, v in character_texts.items()]) + f"\nCharacter: {self.character['name']} responds as {self.character['personality']}:"
        text = self.llm.generate(prompt)
        self.llm.fine_tune({"input": prompt, "output": text}, self.pc_id)
        try:
            requests.post(f"{SERVER_URL}/save_training_data", json={"dataset": {"input": prompt, "output": text}, "pc": self.pc_id, "token": self.token})
        except requests.exceptions.RequestException as e:
            print(f"Error saving training data to server: {e}")
        return text

    def synthesize_audio(self, text):
        speaker_wav = None
        if self.character.get("tts") == "xttsv2" and self.local_reference_audio_path:
            speaker_wav = self.local_reference_audio_path

        # Use a unique ID for audio files to prevent overwriting
        audio_path = f"character_audio_{self.character['name']}_{uuid.uuid4()}.wav"
        self.tts.synthesize(text, audio_path, speaker_wav=speaker_wav)
        return audio_path

@app.post("/character")
async def generate_character(data: dict):
    token = data.get("token")
    if not hasattr(app.state, 'character_client_instance'):
        raise HTTPException(status_code=500, detail="CharacterClient not initialized.")
    client = app.state.character_client_instance
    if client.token != token: # Basic token validation
        raise HTTPException(status_code=401, detail="Invalid token for this client instance.")

    narration = data["narration"]
    character_texts = data["character_texts"]
    response_text = client.generate_response(narration, character_texts)
    audio_path = client.synthesize_audio(response_text)
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    # Encode audio data to base64 for JSON compatibility
    encoded_audio_data = base64.b64encode(audio_data).decode('utf-8')
    return {"text": response_text, "audio_data": encoded_audio_data}

# Background heartbeat task
async def heartbeat_task(client: CharacterClient):
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        client._register_with_server() # Call the synchronous method


@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event.
    Initializes the CharacterClient and starts the heartbeat task.
    """
    # This part is crucial: CharacterClient needs to be initialized once
    # and its instance stored where the FastAPI routes can access it.
    # The token should be passed as a command-line argument when running the client.
    if len(sys.argv) < 2:
        print("Usage: python character_client.py <token>")
        sys.exit(1)
    token = sys.argv[1]
    app.state.character_client_instance = CharacterClient(token)

    # Start the heartbeat task in the background
    asyncio.create_task(heartbeat_task(app.state.character_client_instance))
    print(f"Client {app.state.character_client_instance.pc_id} started and heartbeat initiated.")

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run will automatically call startup_event and shutdown_event
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
