import requests
import secrets
import pygame
import uuid
import os
import base64
from datetime import datetime, timezone, timedelta
import time # For periodic health checks
import threading # For running health checks in background

from .config import CHARACTERS_AUDIO_PATH
from .database import Database

# How often to perform health checks on clients that are 'Online_Heartbeat'
CLIENT_HEALTH_CHECK_INTERVAL_SECONDS = 60 * 2 # Every 2 minutes
# Timeout for the health check request itself
CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS = 5

class ClientManager:
    def __init__(self, db: Database):
        self.db = db
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
                print("Pygame mixer initialized for ClientManager.")
            except pygame.error as e:
                print(f"Warning: Pygame mixer could not be initialized in ClientManager: {e}.")

        self.health_check_thread = None
        self.stop_health_check_event = threading.Event()
        # self.start_periodic_health_checks() # Start when ClientManager is initialized

    def generate_token(self, pc_id: str) -> str:
        token = secrets.token_hex(24)
        # Ensure character exists or is PC1 before saving token
        char = self.db.get_character(pc_id)
        if not char and pc_id == "PC1":
            self.db.save_character(name="ServerChar_PC1", personality="Host", goals="Manage", backstory="Server internal char",
                                   tts="piper", tts_model="en_US-ryan-high", reference_audio_filename=None, pc_id="PC1", llm_model=None)
        elif not char:
             print(f"Warning: Generating token for '{pc_id}' but character does not exist. Create character in UI first.")
             # Optionally, prevent token generation here or let save_client_token handle it if it checks FK.
             # For now, we rely on Gradio to create char first.

        self.db.save_client_token(pc_id, token) # This sets status to 'Registered'
        print(f"Token generated and saved for '{pc_id}'.")
        return token

    def get_clients_for_story_progression(self): # Renamed from get_active_clients_info
        """Retrieves clients that are 'Online_Responsive' and recently seen."""
        return self.db.get_clients_for_story_progression()

    def validate_token(self, pc_id: str, token: str) -> bool:
        client_details = self.db.get_client_token_details(pc_id)
        if client_details and client_details.get('token') == token:
            # Could add further checks here, e.g., if client_details.get('status') == 'Deactivated'
            return True
        # print(f"Token validation failed for {pc_id}.") # Can be noisy
        return False

    def _perform_single_health_check(self, client_info: dict):
        """Performs a health check on a single client and updates its status."""
        pc_id = client_info.get("pc_id")
        ip_address = client_info.get("ip_address")
        client_port = client_info.get("client_port")

        if not all([pc_id, ip_address, client_port]):
            print(f"Health Check: Insufficient info for client {pc_id}. Skipping.")
            return

        health_url = f"http://{ip_address}:{client_port}/health"
        try:
            # print(f"Health Check: Pinging {pc_id} at {health_url}")
            response = requests.get(health_url, timeout=CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            health_data = response.json()

            if health_data.get("status") == "ok":
                self.db.update_client_status(pc_id, "Online_Responsive")
                # print(f"Health Check: Client {pc_id} is responsive. Status: Online_Responsive.")
            elif health_data.get("status") == "degraded":
                self.db.update_client_status(pc_id, "Error_API_Degraded") # New status
                print(f"Health Check: Client {pc_id} API is degraded: {health_data.get('detail')}. Status: Error_API_Degraded.")
            else: # Unexpected status in response
                self.db.update_client_status(pc_id, "Error_API")
                print(f"Health Check: Client {pc_id} returned unexpected health status: {health_data.get('status')}. Status: Error_API.")

        except requests.exceptions.Timeout:
            print(f"Health Check: Timeout pinging {pc_id} at {health_url}. Status: Error_API (Timeout).")
            self.db.update_client_status(pc_id, "Error_API")
        except requests.exceptions.ConnectionError:
            print(f"Health Check: Connection error pinging {pc_id} at {health_url}. Status: Error_Unreachable.")
            self.db.update_client_status(pc_id, "Error_Unreachable")
        except requests.exceptions.RequestException as e:
            print(f"Health Check: Error pinging {pc_id} at {health_url}: {e}. Status: Error_API.")
            self.db.update_client_status(pc_id, "Error_API")
        except Exception as e: # Catch other errors like JSONDecodeError
            print(f"Health Check: Unexpected error during health check for {pc_id}: {e}. Status: Error_API.")
            self.db.update_client_status(pc_id, "Error_API")


    def _periodic_health_check_loop(self):
        """The loop that runs in a separate thread to check client health."""
        print("ClientManager: Periodic health check thread started.")
        while not self.stop_health_check_event.is_set():
            try:
                # Get clients that are 'Online_Heartbeat' or those whose API was previously problematic but might recover
                clients_to_check = self.db.get_all_client_statuses() # Get all, then filter
                if clients_to_check:
                    for client_data in clients_to_check:
                        current_status = client_data.get("status")
                        # Check clients that are just heartbeating, or were in an error state previously
                        if current_status in ["Online_Heartbeat", "Error_API", "Error_API_Degraded", "Error_Unreachable"]:
                            # print(f"Health Check: Evaluating client {client_data.get('pc_id')} with status {current_status}")
                            self._perform_single_health_check(client_data)
                        elif current_status == "Online_Responsive":
                            # For responsive clients, check if last_seen is too old. If so, maybe revert to Online_Heartbeat.
                            last_seen_iso = client_data.get("last_seen")
                            if last_seen_iso:
                                last_seen_dt = datetime.fromisoformat(last_seen_iso)
                                # If last heartbeat was more than, say, 2*heartbeat_interval ago, it's stale.
                                if datetime.now(timezone.utc) - last_seen_dt > timedelta(seconds=HEARTBEAT_INTERVAL_SECONDS * 2.5): # HEARTBEAT_INTERVAL_SECONDS from client
                                    print(f"Health Check: Client {client_data.get('pc_id')} was Online_Responsive but last heartbeat is stale. Setting to Offline.")
                                    self.db.update_client_status(client_data.get('pc_id'), "Offline")


            except Exception as e:
                print(f"ClientManager: Error in periodic health check loop: {e}")

            # Wait for the defined interval or until stop event is set
            self.stop_health_check_event.wait(CLIENT_HEALTH_CHECK_INTERVAL_SECONDS)
        print("ClientManager: Periodic health check thread stopped.")

    def start_periodic_health_checks(self):
        if self.health_check_thread is None or not self.health_check_thread.is_alive():
            self.stop_health_check_event.clear()
            self.health_check_thread = threading.Thread(target=self._periodic_health_check_loop, daemon=True)
            self.health_check_thread.start()
        else:
            print("ClientManager: Health check thread already running.")

    def stop_periodic_health_checks(self):
        print("ClientManager: Stopping periodic health checks...")
        self.stop_health_check_event.set()
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5) # Wait for thread to finish
        print("ClientManager: Periodic health checks stopped.")


    def send_to_client(self, client_pc_id: str, client_ip: str, client_port: int, narration: str, character_texts: dict) -> str:
        character = self.db.get_character(client_pc_id)
        if not character:
            print(f"No character data for client {client_pc_id}. Cannot send.")
            self.db.update_client_status(client_pc_id, "Error_API", datetime.now(timezone.utc).isoformat()) # Or a more specific error
            return ""

        token = self.db.get_token(client_pc_id)
        if not token:
            print(f"No token for client {client_pc_id}. Cannot send.")
            self.db.update_client_status(client_pc_id, "Error_API", datetime.now(timezone.utc).isoformat())
            return ""

        url = f"http://{client_ip}:{client_port}/character"
        # print(f"Sending data to client {client_pc_id} at {url}") # Can be verbose

        request_payload = {"narration": narration, "character_texts": character_texts, "token": token}

        # Retry logic for sending data to client
        max_retries = 2 # Example: try original + 2 retries
        base_delay = 1  # seconds

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, json=request_payload, timeout=15)
                response.raise_for_status() # Raises HTTPError for 4xx/5xx responses
                response_data = response.json()
                client_text_response = response_data.get("text")
            encoded_audio_data = response_data.get("audio_data")

            if encoded_audio_data and pygame.mixer.get_init():
                sane_char_name = "".join(c if c.isalnum() else "_" for c in character.get('name', client_pc_id))
                audio_dir = os.path.join(CHARACTERS_AUDIO_PATH, sane_char_name)
                os.makedirs(audio_dir, exist_ok=True)
                audio_filename = f"{client_pc_id}_{uuid.uuid4()}.wav"
                audio_path = os.path.join(audio_dir, audio_filename)
                decoded_audio_data = base64.b64decode(encoded_audio_data)
                with open(audio_path, "wb") as f: f.write(decoded_audio_data)

                pygame.mixer.Sound(audio_path).play()
                # print(f"Played audio from {client_pc_id}") # Can be verbose

            # On successful interaction, confirm client is responsive
            self.db.update_client_status(client_pc_id, "Online_Responsive", datetime.now(timezone.utc).isoformat())
            return client_text_response # Success, exit retry loop

            except requests.exceptions.Timeout:
                print(f"Attempt {attempt + 1}/{max_retries + 1}: Timeout sending to {client_pc_id} at {url}.")
                if attempt == max_retries:
                    self.db.update_client_status(client_pc_id, "Error_API", datetime.now(timezone.utc).isoformat())
                    return "" # Final attempt failed
            except requests.exceptions.ConnectionError:
                print(f"Attempt {attempt + 1}/{max_retries + 1}: Connection error with {client_pc_id} at {url}.")
                if attempt == max_retries:
                    self.db.update_client_status(client_pc_id, "Error_Unreachable", datetime.now(timezone.utc).isoformat())
                    return "" # Final attempt failed
            except requests.exceptions.RequestException as e: # Includes HTTPError from raise_for_status
                print(f"Attempt {attempt + 1}/{max_retries + 1}: Request error with {client_pc_id} at {url}: {e}")
                if attempt == max_retries:
                    self.db.update_client_status(client_pc_id, "Error_API", datetime.now(timezone.utc).isoformat())
                    return "" # Final attempt failed
            except Exception as e: # Catch other errors like JSONDecodeError within the loop
                print(f"Attempt {attempt + 1}/{max_retries + 1}: Unexpected error processing response from {client_pc_id} at {url}: {e}")
                if attempt == max_retries:
                    self.db.update_client_status(client_pc_id, "Error_API", datetime.now(timezone.utc).isoformat())
                    return "" # Final attempt failed

            # If not the last attempt, wait before retrying
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) # Exponential backoff
                print(f"Waiting {delay}s before retry...")
                time.sleep(delay)

        # This part should ideally not be reached if logic is correct, means loop finished without returning
        print(f"send_to_client for {client_pc_id} failed after all retries.")
        self.db.update_client_status(client_pc_id, "Error_API", datetime.now(timezone.utc).isoformat())
        return ""

    def deactivate_client_pc(self, pc_id: str):
        self.db.update_client_status(pc_id, "Deactivated", datetime.now(timezone.utc).isoformat())
        print(f"Client {pc_id} marked as Deactivated.")

    # Ensure to stop the thread on application shutdown if ClientManager instance is long-lived
    # This might be handled by the main application's shutdown sequence.
    def __del__(self):
        self.stop_periodic_health_checks()
