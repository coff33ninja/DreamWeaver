import requests
import secrets
import pygame
import uuid
import os
import base64
from datetime import datetime, timezone, timedelta
import threading
import asyncio # Added asyncio

from .config import CHARACTERS_AUDIO_PATH
from .database import Database


CLIENT_HEALTH_CHECK_INTERVAL_SECONDS = 60 * 2
CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS = 5
# Retry settings for send_to_client
SEND_TO_CLIENT_MAX_RETRIES = 2
SEND_TO_CLIENT_BASE_DELAY_SECONDS = 1
SEND_TO_CLIENT_REQUEST_TIMEOUT_SECONDS = 15 # Timeout for the actual /character request


class ClientManager:
    def __init__(self, db: Database):
        """
        Initialize the ClientManager with a database connection and prepare audio playback and health check mechanisms.
        
        Attempts to initialize the pygame mixer for audio playback. Sets up threading constructs for managing periodic client health checks.
        """
        self.db = db
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except pygame.error as e:
                print(f"ClientManager: Warning - Pygame mixer could not be initialized: {e}.")

        self.health_check_thread = None
        self.stop_health_check_event = threading.Event()
        # Consider starting health checks from CSM or main app to ensure DB is fully ready.

    def generate_token(self, Actor_id: str) -> str:
        """
        Generate and store a unique authentication token for the specified character.
        
        If the character does not exist and the Actor_id is "Actor1", creates a default server character before generating the token. The generated token is saved in the database and returned.
        
        Parameters:
            Actor_id (str): The unique identifier of the character.
        
        Returns:
            str: The generated authentication token.
        """
        token = secrets.token_hex(24)
        char = self.db.get_character(Actor_id)
        if not char and Actor_id == "Actor1": # Should Actor1 even have a token generated this way?
            self.db.save_character(name="ServerChar_Actor1", personality="Host", goals="Manage", backstory="Server internal char",
                                   tts="piper", tts_model="en_US-ryan-high", reference_audio_filename=None, Actor_id="Actor1", llm_model=None)
        elif not char:
             print(f"ClientManager: Warning - Generating token for '{Actor_id}' but character does not exist.")
        self.db.save_client_token(Actor_id, token)
        return token

    def get_clients_for_story_progression(self):
        """
        Retrieve a list of clients eligible for story progression from the database.
        
        Returns:
            List of client records that are currently eligible to participate in story progression.
        """
        return self.db.get_clients_for_story_progression()

    def validate_token(self, Actor_id: str, token: str) -> bool:
        """
        Validate whether the provided token matches the stored token for the given Actor and that the client is not deactivated.
        
        Returns:
            bool: True if the token is valid and the client is active; otherwise, False.
        """
        client_details = self.db.get_client_token_details(Actor_id)
        return bool(client_details and client_details.get('token') == token and client_details.get('status') != 'Deactivated')

    def _perform_single_health_check_blocking(self, client_info: dict):
<<<<<<< HEAD
        """Blocking /health check and DB-status update."""
=======
        """
        Performs a blocking health check on a client by querying its `/health` endpoint and updates the client's status in the database based on the response.
        
        Parameters:
            client_info (dict): Dictionary containing client connection details, including "Actor_id", "ip_address", and "client_port".
        """
>>>>>>> 96fc77cc2cce22f1a9028bf0e9399df2f81b2e3d
        Actor_id = client_info.get("Actor_id")
        ip_address = client_info.get("ip_address")
        client_port = client_info.get("client_port")

        if not all([Actor_id, ip_address, client_port]):
            return

        health_url = f"http://{ip_address}:{client_port}/health"
        new_status = "Error_API" # Default to error if checks fail
        try:
            response = requests.get(health_url, timeout=CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            health_data = response.json()

            if health_data.get("status") == "ok":
                new_status = "Online_Responsive"
            elif health_data.get("status") == "degraded":
                new_status = "Error_API_Degraded"
            # else:
            #     print(f"Health Check ({Actor_id}): Unexpected health status '{health_data.get('status')}'")

        except requests.exceptions.Timeout:
            # print(f"Health Check ({Actor_id}): Timeout at {health_url}")
            new_status = "Error_API" # Or more specific timeout status
        except requests.exceptions.ConnectionError:
            # print(f"Health Check ({Actor_id}): Connection error at {health_url}")
            new_status = "Error_Unreachable"
        except requests.exceptions.RequestException:
            # print(f"Health Check ({Actor_id}): Request error at {health_url}")
            new_status = "Error_API"
        except Exception:
            # print(f"Health Check ({Actor_id}): Unexpected error")
            new_status = "Error_API"

        self.db.update_client_status(Actor_id, new_status)

    def _periodic_health_check_loop(self):
        """
        Continuously monitors client health statuses and updates them based on responsiveness.
        
        This loop runs in a background thread, periodically checking each client's status. It performs health checks for clients in certain error or heartbeat states, and marks clients as offline if their last heartbeat is stale. The loop continues until signaled to stop.
        """
        print("ClientManager: Periodic health check thread started.")
        while not self.stop_health_check_event.is_set():
            try:
                clients_to_check = self.db.get_all_client_statuses()
                if clients_to_check:
                    for client_data in clients_to_check:
                        current_status = client_data.get("status")
                        if current_status in ["Online_Heartbeat", "Error_API", "Error_API_Degraded", "Error_Unreachable"]:
                            self._perform_single_health_check_blocking(client_data)
                        elif current_status == "Online_Responsive":
                            last_seen_iso = client_data.get("last_seen")
                            if last_seen_iso:
                                last_seen_dt = datetime.fromisoformat(last_seen_iso)
                                if datetime.now(timezone.utc) - last_seen_dt > timedelta(seconds=CLIENT_HEALTH_CHECK_INTERVAL_SECONDS * 2.5):
                                    print(f"Health Check: Client {client_data.get('Actor_id')} unresponsive (stale heartbeat). Status: Offline.")
                                    self.db.update_client_status(client_data.get('Actor_id'), "Offline")
            except Exception as e:
                print(f"ClientManager: Error in health check loop: {e}")
            self.stop_health_check_event.wait(CLIENT_HEALTH_CHECK_INTERVAL_SECONDS)
        print("ClientManager: Periodic health check thread stopped.")

    def start_periodic_health_checks(self):
        """
        Start the background thread that periodically checks the health status of all clients.
        
        If the health check thread is already running, this method does nothing.
        """
        if self.health_check_thread is None or not self.health_check_thread.is_alive():
            self.stop_health_check_event.clear()
            self.health_check_thread = threading.Thread(target=self._periodic_health_check_loop, daemon=True)
            self.health_check_thread.start()

    def stop_periodic_health_checks(self):
        """
        Stop the background thread responsible for periodic client health checks.
        
        Signals the health check loop to terminate and waits for the thread to finish execution.
        """
        self.stop_health_check_event.set()
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=max(1, CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS + 1))


    async def send_to_client(self, client_Actor_id: str, client_ip: str, client_port: int, narration: str, character_texts: dict) -> str:
        """
        Asynchronously sends narration and character texts to a client, handling retries, audio playback, and client status updates.
        
        Attempts to deliver narration and character-specific texts to the specified client via HTTP POST, using the client's token for authentication. On a successful response, processes any returned text and base64-encoded audio data, saving and playing the audio if available. Updates the client's status in the database based on the outcome. Retries the request with exponential backoff on failure, and returns an empty string if all attempts fail.
        
        Parameters:
            client_Actor_id (str): The unique identifier of the client character.
            client_ip (str): The IP address of the client.
            client_port (int): The port number of the client.
            narration (str): The narration text to send.
            character_texts (dict): A mapping of character names to their respective texts.
        
        Returns:
            str: The text response from the client, or an empty string if the request fails.
        """
        character = self.db.get_character(client_Actor_id) # Blocking DB call
        if not character:
            print(f"send_to_client: No character data for {client_Actor_id}.")
            self.db.update_client_status(client_Actor_id, "Error_API")
            return ""

        token = self.db.get_token(client_Actor_id) # Blocking DB call
        if not token:
            print(f"send_to_client: No token for {client_Actor_id}.")
            self.db.update_client_status(client_Actor_id, "Error_API")
            return ""

        url = f"http://{client_ip}:{client_port}/character"
        request_payload = {"narration": narration, "character_texts": character_texts, "token": token}

        def _blocking_post_request():
<<<<<<< HEAD
            """POST to client with timeout."""
=======
            """
            Send a blocking HTTP POST request with the specified payload and timeout.
            
            Returns:
                Response: The HTTP response object from the POST request.
            """
>>>>>>> 96fc77cc2cce22f1a9028bf0e9399df2f81b2e3d
            return requests.post(url, json=request_payload, timeout=SEND_TO_CLIENT_REQUEST_TIMEOUT_SECONDS)

        for attempt in range(SEND_TO_CLIENT_MAX_RETRIES + 1):
            try:
                # print(f"send_to_client (Attempt {attempt+1}): Sending to {client_Actor_id} at {url}") # Verbose
                response = await asyncio.to_thread(_blocking_post_request)
                response.raise_for_status()
                response_data = response.json()
                client_text_response = response_data.get("text")
                encoded_audio_data = response_data.get("audio_data")

                if encoded_audio_data and pygame.mixer.get_init():
                    # This part (decode, save, play) is also blocking
                    def _handle_audio():
<<<<<<< HEAD
                        """Decode, save, and play audio response."""
=======
                        """
                        Decodes base64-encoded audio data, saves it as a WAV file in a character-specific directory, and plays the audio using pygame.
                        
                        The audio file is named using the client Actor ID and a unique identifier to avoid collisions.
                        """
>>>>>>> 96fc77cc2cce22f1a9028bf0e9399df2f81b2e3d
                        sane_char_name = "".join(c if c.isalnum() else "_" for c in character.get('name', client_Actor_id))
                        audio_dir = os.path.join(CHARACTERS_AUDIO_PATH, sane_char_name)
                        os.makedirs(audio_dir, exist_ok=True)
                        audio_filename = f"{client_Actor_id}_{uuid.uuid4()}.wav"
                        audio_path = os.path.join(audio_dir, audio_filename)
                        decoded_audio_data = base64.b64decode(encoded_audio_data)
                        with open(audio_path, "wb") as f:
                            f.write(decoded_audio_data)
                        pygame.mixer.Sound(audio_path).play()
                    await asyncio.to_thread(_handle_audio)

                self.db.update_client_status(client_Actor_id, "Online_Responsive") # Blocking DB call
                return client_text_response

            except requests.exceptions.Timeout:
                print(f"send_to_client (Attempt {attempt+1}): Timeout for {client_Actor_id} at {url}.")
                if attempt == SEND_TO_CLIENT_MAX_RETRIES:
                    self.db.update_client_status(client_Actor_id, "Error_API") # Blocking
            except requests.exceptions.ConnectionError:
                print(f"send_to_client (Attempt {attempt+1}): Connection error for {client_Actor_id} at {url}.")
                if attempt == SEND_TO_CLIENT_MAX_RETRIES:
                    self.db.update_client_status(client_Actor_id, "Error_Unreachable") # Blocking
            except requests.exceptions.RequestException as e:
                print(f"send_to_client (Attempt {attempt+1}): Request error for {client_Actor_id} at {url}: {e}")
                if attempt == SEND_TO_CLIENT_MAX_RETRIES:
                    self.db.update_client_status(client_Actor_id, "Error_API") # Blocking
            except Exception as e:
                print(f"send_to_client (Attempt {attempt+1}): Unexpected error for {client_Actor_id}: {e}")
                if attempt == SEND_TO_CLIENT_MAX_RETRIES:
                    self.db.update_client_status(client_Actor_id, "Error_API") # Blocking

            if attempt < SEND_TO_CLIENT_MAX_RETRIES:
                delay = SEND_TO_CLIENT_BASE_DELAY_SECONDS * (2 ** attempt)
                # print(f"send_to_client: Waiting {delay}s before retry for {client_Actor_id}...") # Verbose
                await asyncio.sleep(delay) # Use asyncio.sleep for async context

        print(f"send_to_client for {client_Actor_id} failed after all retries.")
        return "" # Return empty if all retries fail

    def deactivate_client_Actor(self, Actor_id: str): # Blocking DB call
        """
        Mark the specified client as deactivated in the database.
        
        Parameters:
            Actor_id (str): The identifier of the client to deactivate.
        """
        self.db.update_client_status(Actor_id, "Deactivated")
        print(f"Client {Actor_id} marked as Deactivated.")

    def __del__(self):
        """
        Ensures that periodic health checks are stopped when the ClientManager instance is destroyed.
        """
        self.stop_periodic_health_checks()
