import requests
import secrets
import pygame
import uuid
import os
import base64

class ClientManager:
    def __init__(self, db):
        self.db = db
        # The base URL no longer needs a format placeholder
        if not pygame.mixer.get_init():
            pygame.mixer.init()

    def generate_token(self, client_id):
        token = secrets.token_hex(16)
        self.db.save_token(client_id, token)
        return token

    def get_active_clients(self):
        return self.db.get_active_clients()

    def validate_token(self, pc: str, token: str) -> bool:
        """Validates if the provided token matches the one stored for the PC."""
        stored_token = self.db.get_token(pc)
        return stored_token == token

    def send_to_client(self, client_id, client_ip, narration, character_texts):
        character = self.db.get_character(client_id)
        if not character:
            return ""
        token = self.db.get_token(client_id)
        url = f"http://{client_ip}:8000/character" # Construct URL with registered IP
        try:
            response = requests.post(url, json={"narration": narration, "character_texts": character_texts, "token": token}, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            audio_dir = f"E:/DreamWeaver/data/audio/characters/{character['name']}"
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f"{uuid.uuid4()}.wav")
            with open(audio_path, "wb") as f:
                # Decode the base64 string back to bytes before writing
                decoded_audio_data = base64.b64decode(response.json()["audio_data"])
                f.write(decoded_audio_data)
            pygame.mixer.Sound(audio_path).play() # Play client's audio on server
            return response.json()["text"] # Return client's text response
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with client {client_id} at {url}: {e}")
            return ""
        except Exception as e:
            print(f"An unexpected error occurred processing client {client_id} response: {e}")
            return ""
