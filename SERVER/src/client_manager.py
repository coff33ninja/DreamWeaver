import requests
import secrets
import pygame # For audio playback on server from client
import uuid
import os
import base64
from datetime import datetime, timezone

from .config import CHARACTERS_AUDIO_PATH # Path for storing character audio received from clients
from .database import Database # Type hinting and potentially direct use if needed

class ClientManager:
    def __init__(self, db: Database): # Type hint for db
        self.db = db
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
                print("Pygame mixer initialized for ClientManager.")
            except pygame.error as e:
                print(f"Warning: Pygame mixer could not be initialized in ClientManager: {e}. Audio playback from clients will fail.")


    def generate_token(self, pc_id: str) -> str:
        """Generates a new unique token for a client and saves it."""
        token = secrets.token_hex(24) # Increased token length
        # The database method `save_client_token` handles insert/update.
        # It also ensures the character entry exists or creates a placeholder if pc_id is "PC1".
        # For other pc_ids, character should be created first via Gradio.
        # We might need to ensure a character placeholder exists before token generation if not PC1.
        char_exists = self.db.get_character(pc_id)
        if not char_exists and pc_id != "PC1":
             # This logic might be better in Gradio: ensure char exists before allowing token gen.
             print(f"Warning: Generating token for {pc_id} but character does not exist. Create character first.")
             # Or, create a placeholder character:
             # self.db.save_character(name=pc_id, personality="Default", goals="", backstory="", tts="piper", tts_model="en_US-ryan-high", reference_audio_filename=None, pc_id=pc_id)

        self.db.save_client_token(pc_id, token)
        print(f"Token generated and saved for {pc_id}.")
        return token

    def get_active_clients_info(self): # Renamed from get_active_clients
        """Retrieves information (pc_id, ip_address, client_port) for active clients."""
        # Logic for "active" (e.g., seen in last X minutes) is now in db.get_active_clients()
        return self.db.get_active_clients() # This method now returns dicts with port

    def validate_token(self, pc_id: str, token: str) -> bool:
        """Validates if the provided token matches the one stored and active for the PC_ID."""
        client_details = self.db.get_client_token_details(pc_id) # Fetches token and status
        if client_details and client_details.get('token') == token:
            # Could also check status here if needed, e.g., if status is 'Disabled'
            return True
        print(f"Token validation failed for {pc_id}.")
        return False

    def send_to_client(self, client_pc_id: str, client_ip: str, client_port: int, narration: str, character_texts: dict) -> str:
        """
        Sends narration and context to a specific client and gets a response.
        Uses the client_port registered by the client.
        """
        character = self.db.get_character(client_pc_id)
        if not character:
            print(f"No character data found for client {client_pc_id}. Cannot send message.")
            self.db.update_client_status(client_pc_id, "Error", datetime.now(timezone.utc).isoformat())
            return "" # Return empty string or raise an error

        token = self.db.get_token(client_pc_id) # Get token for the client
        if not token:
            print(f"No token found for client {client_pc_id}. Cannot authenticate message.")
            self.db.update_client_status(client_pc_id, "Error", datetime.now(timezone.utc).isoformat())
            return ""

        # Construct URL using registered IP and Port
        url = f"http://{client_ip}:{client_port}/character"
        print(f"Sending data to client {client_pc_id} at {url}")

        request_payload = {
            "narration": narration,
            "character_texts": character_texts,
            "token": token # Send the client's own token for it to validate if it wishes
        }

        try:
            response = requests.post(url, json=request_payload, timeout=15) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            client_text_response = response_data.get("text")
            encoded_audio_data = response_data.get("audio_data")

            if encoded_audio_data and pygame.mixer.get_init():
                # Use path from config for storing received audio
                # Ensure character name is filesystem-safe
                sane_char_name = "".join(c if c.isalnum() else "_" for c in character.get('name', client_pc_id))
                audio_dir = os.path.join(CHARACTERS_AUDIO_PATH, sane_char_name)
                os.makedirs(audio_dir, exist_ok=True)

                audio_filename = f"{client_pc_id}_{uuid.uuid4()}.wav"
                audio_path = os.path.join(audio_dir, audio_filename)

                decoded_audio_data = base64.b64decode(encoded_audio_data)
                with open(audio_path, "wb") as f:
                    f.write(decoded_audio_data)

                # Play client's audio on the server
                sound = pygame.mixer.Sound(audio_path)
                sound.play()
                print(f"Played audio from {client_pc_id} received from {url}")
            elif not pygame.mixer.get_init():
                 print(f"Warning: Pygame mixer not initialized. Cannot play audio from client {client_pc_id}.")


            # Update client status to Online on successful communication
            self.db.update_client_status(client_pc_id, "Online", datetime.now(timezone.utc).isoformat())
            return client_text_response # Return client's text response

        except requests.exceptions.Timeout:
            print(f"Timeout error communicating with client {client_pc_id} at {url}.")
            self.db.update_client_status(client_pc_id, "Offline", datetime.now(timezone.utc).isoformat())
            return ""
        except requests.exceptions.ConnectionError:
            print(f"Connection error with client {client_pc_id} at {url}. Is client running and accessible?")
            self.db.update_client_status(client_pc_id, "Offline", datetime.now(timezone.utc).isoformat())
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with client {client_pc_id} at {url}: {e}")
            self.db.update_client_status(client_pc_id, "Error", datetime.now(timezone.utc).isoformat())
            return ""
        except KeyError as e:
            print(f"Key error in response from client {client_pc_id}: {e}. Response: {response.text if 'response' in locals() else 'N/A'}")
            self.db.update_client_status(client_pc_id, "Error", datetime.now(timezone.utc).isoformat())
            return ""
        except Exception as e: # Catch-all for other unexpected errors
            print(f"An unexpected error occurred processing client {client_pc_id} response from {url}: {e}")
            self.db.update_client_status(client_pc_id, "Error", datetime.now(timezone.utc).isoformat())
            return ""

    # Other methods like managing client lifecycle (e.g., explicit deactivation) could go here.
    def deactivate_client_pc(self, pc_id: str):
        """Marks a client as inactive in the database."""
        self.db.update_client_status(pc_id, "Offline_Deactivated", datetime.now(timezone.utc).isoformat())
        print(f"Client {pc_id} marked as deactivated.")
