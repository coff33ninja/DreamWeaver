import requests
import secrets
import pygame
import uuid
import os
import base64
from datetime import datetime, timezone, timedelta
import threading
import asyncio  # Added asyncio
import logging
from typing import Any

from .config import CHARACTERS_AUDIO_PATH
from .database import Database

logger = logging.getLogger("dreamweaver_server")

CLIENT_HEALTH_CHECK_INTERVAL_SECONDS = 60 * 2
CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS = 5
# Retry settings for send_to_client
SEND_TO_CLIENT_MAX_RETRIES = 2
SEND_TO_CLIENT_BASE_DELAY_SECONDS = 1
SEND_TO_CLIENT_REQUEST_TIMEOUT_SECONDS = 15  # Timeout for the actual /character request


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
                logger.warning(
                    f"ClientManager: Pygame mixer could not be initialized: {e}.",
                    exc_info=True,
                )
        self.health_check_thread = None
        self.stop_health_check_event = threading.Event()
        self.active_challenges: dict[str, dict[str, Any]] = (
            {}
        )  # Actor_id -> {"challenge": str, "timestamp": datetime}
        self.CHALLENGE_EXPIRY_SECONDS = 60
        self.CHALLENGE_EXPIRY_SECONDS = 60

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
        if (
            not char and Actor_id == "Actor1"
        ):  # Should Actor1 even have a token generated this way?
            logger.info(
                "Actor1 character not found, creating default server character for token generation."
            )
            self.db.save_character(
                name="ServerChar_Actor1",
                personality="Host",
                goals="Manage",
                backstory="Server internal char",
                tts="piper",
                tts_model="en_US-ryan-high",
                reference_audio_filename=None,
                Actor_id="Actor1",
                llm_model=None,
            )
        elif not char:
            logger.warning(
                f"ClientManager: Generating token for '{Actor_id}' but character does not exist in DB."
            )
        self.db.save_client_token(
            Actor_id, token
        )  # This also clears any existing session token
        logger.info(f"Generated and saved primary token for Actor_id: {Actor_id}")
        return token

    def generate_handshake_challenge(self, Actor_id: str) -> str | None:
        """
        Generates a new handshake challenge for the given Actor_id, stores it with a timestamp,
        and returns the challenge string.
        """
        challenge = secrets.token_urlsafe(32)
        self.active_challenges[Actor_id] = {
            "challenge": challenge,
            "timestamp": datetime.now(timezone.utc),
        }
        logger.info(f"Generated handshake challenge for Actor_id: {Actor_id}")
        return challenge

    def get_and_validate_challenge(self, Actor_id: str) -> str | None:
        """
        Retrieves the stored challenge for an Actor_id if it exists and hasn't expired.
        Removes the challenge after retrieval if it's valid.
        """
        challenge_data = self.active_challenges.get(Actor_id)
        if not challenge_data:
            logger.warning(
                f"No active handshake challenge found for Actor_id: {Actor_id}"
            )
            return None

        issue_time = challenge_data["timestamp"]
        if datetime.now(timezone.utc) - issue_time > timedelta(
            seconds=self.CHALLENGE_EXPIRY_SECONDS
        ):
            logger.warning(
                f"Handshake challenge expired for Actor_id: {Actor_id}. Issued at: {issue_time}"
            )
            del self.active_challenges[Actor_id]  # Clean up expired challenge
            return None

        # Challenge is valid and will be consumed now
        # del self.active_challenges[Actor_id] # Challenge should be deleted only after successful response verification
        return challenge_data["challenge"]

    def clear_challenge(self, Actor_id: str):
        """Removes a challenge for an Actor_id, typically after use or expiry."""
        if Actor_id in self.active_challenges:
            del self.active_challenges[Actor_id]
            logger.info(f"Cleared handshake challenge for Actor_id: {Actor_id}")

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
        is_valid = bool(
            client_details
            and client_details.get("token") == token
            and client_details.get("status") != "Deactivated"
        )
        if not is_valid:
            logger.warning(
                f"Primary token validation failed for Actor_id: {Actor_id}. Provided token: {'***' if token else 'None'}."
            )
        return is_valid

    def authenticate_request_token(self, Actor_id: str, provided_token: str) -> bool:
        """
        Authenticates a request by checking if the provided token is either a valid,
        non-expired session token or the correct primary token for the Actor_id.
        """
        if not Actor_id or not provided_token:
            logger.warning("Authentication attempt with missing Actor_id or token.")
            return False

        client_details = self.db.get_client_token_details(Actor_id)
        if not client_details:
            logger.warning(
                f"No client details found for Actor_id: {Actor_id} during token authentication."
            )
            return False

        # 1. Check for valid session token
        session_token = client_details.get("session_token")
        session_token_expiry_iso = client_details.get("session_token_expiry")

        if session_token and session_token == provided_token:
            if session_token_expiry_iso:
                try:
                    expiry_dt = datetime.fromisoformat(session_token_expiry_iso)
                    if datetime.now(timezone.utc) < expiry_dt:
                        logger.debug(
                            f"Authenticated Actor_id: {Actor_id} using valid session token."
                        )
                        return True
                    else:
                        logger.warning(
                            f"Session token expired for Actor_id: {Actor_id}. Expiry: {session_token_expiry_iso}. Clearing it."
                        )
                        # Clear the expired session token from DB
                        self.db.update_client_session_token(Actor_id, None, None)
                        # Do not proceed to check primary token if an expired session token was provided.
                        # Client should re-handshake.
                        return False
                except ValueError:
                    logger.error(
                        f"Invalid session_token_expiry format for Actor_id: {Actor_id}: {session_token_expiry_iso}",
                        exc_info=True,
                    )
                    # Treat as expired or invalid session token
                    return False  # Or perhaps clear it and then check primary, but safer to fail here.
            else:
                # Session token exists but no expiry - treat as invalid or an issue.
                logger.warning(
                    f"Session token found for Actor_id: {Actor_id} but has no expiry. Treating as invalid."
                )
                return False  # Safer to require expiry

        # 2. If not a valid session token, check primary token
        primary_token = client_details.get("token")
        if primary_token and primary_token == provided_token:
            # This means the client is using its primary token.
            # This could be allowed, or we might want to enforce session tokens after first handshake.
            # For now, allowing primary token.
            logger.debug(f"Authenticated Actor_id: {Actor_id} using primary token.")
            return True

        logger.warning(
            f"Authentication failed for Actor_id: {Actor_id}. Provided token did not match session or primary token."
        )
        return False

    def _perform_single_health_check_blocking(self, client_info: dict):
        """
        Performs a blocking health check on a client by querying its `/health` endpoint and updates the client's status in the database based on the response.

        Parameters:
            client_info (dict): Dictionary containing client connection details, including "Actor_id", "ip_address", and "client_port".
        """
        Actor_id = client_info.get("Actor_id")
        ip_address = client_info.get("ip_address")
        client_port = client_info.get("client_port")

        if not all([Actor_id, ip_address, client_port]):
            logger.warning(
                f"Attempted health check with incomplete client_info: {client_info}"
            )
            return

        health_url = f"http://{ip_address}:{client_port}/health"
        new_status = "Error_API"  # Default to error if checks fail
        try:
            # logger.debug(f"Performing health check for {Actor_id} at {health_url}")
            response = requests.get(
                health_url, timeout=CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            health_data = response.json()

            if health_data.get("status") == "ok":
                new_status = "Online_Responsive"
            elif health_data.get("status") == "degraded":
                new_status = "Error_API_Degraded"
                logger.warning(
                    f"Health Check ({Actor_id}): Client reported degraded status. Health data: {health_data}"
                )
            else:
                logger.warning(
                    f"Health Check ({Actor_id}): Unexpected health status '{health_data.get('status')}' from {health_url}. Data: {health_data}"
                )

        except requests.exceptions.Timeout:
            logger.warning(f"Health Check ({Actor_id}): Timeout at {health_url}")
            new_status = "Error_API"
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Health Check ({Actor_id}): Connection error at {health_url}"
            )
            new_status = "Error_Unreachable"
        except requests.exceptions.RequestException as e:  # Includes HTTPError
            logger.warning(
                f"Health Check ({Actor_id}): Request error at {health_url}: {e}"
            )
            new_status = "Error_API"
        except Exception as e:  # Catch any other unexpected error, like JSONDecodeError
            logger.error(
                f"Health Check ({Actor_id}): Unexpected error during health check for {health_url}: {e}",
                exc_info=True,
            )
            new_status = "Error_API"

        # logger.debug(f"Health check for {Actor_id} completed. New status: {new_status}")
        self.db.update_client_status(Actor_id, new_status)

    def _periodic_health_check_loop(self):
        """
        Continuously monitors client health statuses and updates them based on responsiveness.

        This loop runs in a background thread, periodically checking each client's status. It performs health checks for clients in certain error or heartbeat states, and marks clients as offline if their last heartbeat is stale. The loop continues until signaled to stop.
        """
        logger.info("ClientManager: Periodic health check thread started.")
        while not self.stop_health_check_event.is_set():
            try:
                clients_to_check = self.db.get_all_client_statuses()
                if clients_to_check:
                    for client_data in clients_to_check:
                        if self.stop_health_check_event.is_set():
                            break  # Exit early if stopped

                        current_status = client_data.get("status")
                        actor_id = client_data.get("Actor_id")
                        # logger.debug(f"Health Check Loop: Evaluating client {actor_id} with status {current_status}")

                        if current_status in [
                            "Online_Heartbeat",
                            "Error_API",
                            "Error_API_Degraded",
                            "Error_Unreachable",
                        ]:
                            # logger.debug(f"Health Check Loop: Performing direct health check for {actor_id} (Status: {current_status})")
                            self._perform_single_health_check_blocking(client_data)
                        elif current_status == "Online_Responsive":
                            last_seen_iso = client_data.get("last_seen")
                            if last_seen_iso:
                                try:
                                    last_seen_dt = datetime.fromisoformat(last_seen_iso)
                                    if datetime.now(
                                        timezone.utc
                                    ) - last_seen_dt > timedelta(
                                        seconds=CLIENT_HEALTH_CHECK_INTERVAL_SECONDS
                                        * 2.5
                                    ):
                                        logger.warning(
                                            f"Health Check: Client {actor_id} unresponsive (stale heartbeat while Online_Responsive). Marking Offline."
                                        )
                                        self.db.update_client_status(
                                            actor_id, "Offline"
                                        )
                                except ValueError:
                                    logger.error(
                                        f"Health Check: Could not parse last_seen timestamp '{last_seen_iso}' for client {actor_id}",
                                        exc_info=True,
                                    )
            except Exception as e:
                logger.error(
                    f"ClientManager: Error in health check loop: {e}", exc_info=True
                )

            # Wait for the interval or until stop event is set
            # Use wait with a timeout to make the loop check self.stop_health_check_event more frequently
            # than CLIENT_HEALTH_CHECK_INTERVAL_SECONDS if needed, but not busy-wait.
            self.stop_health_check_event.wait(CLIENT_HEALTH_CHECK_INTERVAL_SECONDS)
        logger.info("ClientManager: Periodic health check thread stopped.")

    def start_periodic_health_checks(self):
        """
        Start the background thread that periodically checks the health status of all clients.

        If the health check thread is already running, this method does nothing.
        """
        if self.health_check_thread is None or not self.health_check_thread.is_alive():
            self.stop_health_check_event.clear()
            self.health_check_thread = threading.Thread(
                target=self._periodic_health_check_loop, daemon=True
            )
            self.health_check_thread.start()
            logger.info("Periodic health check thread initiated.")
        else:
            logger.info("Periodic health check thread already running.")

    def stop_periodic_health_checks(self):
        """
        Stop the background thread responsible for periodic client health checks.

        Signals the health check loop to terminate and waits for the thread to finish execution.
        """
        logger.info("Stopping periodic health checks...")
        self.stop_health_check_event.set()
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(
                timeout=max(1, CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS + 1)
            )
            if self.health_check_thread.is_alive():
                logger.warning("Health check thread did not join in time.")
            else:
                logger.info("Health check thread joined successfully.")
        else:
            logger.info("Health check thread was not running or already stopped.")

    async def send_to_client(
        self,
        client_Actor_id: str,
        client_ip: str,
        client_port: int,
        narration: str,
        character_texts: dict,
    ) -> str:
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
        character = self.db.get_character(client_Actor_id)  # Blocking DB call
        if not character:
            logger.warning(
                f"send_to_client: No character data for {client_Actor_id}. Cannot send."
            )
            self.db.update_client_status(client_Actor_id, "Error_API")
            return ""

        token = self.db.get_token(client_Actor_id)  # Blocking DB call
        if not token:
            logger.warning(
                f"send_to_client: No token for {client_Actor_id}. Cannot send."
            )
            self.db.update_client_status(client_Actor_id, "Error_API")
            return ""

        url = f"http://{client_ip}:{client_port}/character"
        request_payload = {
            "narration": narration,
            "character_texts": character_texts,
            "token": token,
        }

        def _blocking_post_request():
            """
            Send a blocking HTTP POST request with the specified payload and timeout.

            Returns:
                Response: The HTTP response object from the POST request.
            """
            return requests.post(
                url,
                json=request_payload,
                timeout=SEND_TO_CLIENT_REQUEST_TIMEOUT_SECONDS,
            )

        for attempt in range(SEND_TO_CLIENT_MAX_RETRIES + 1):
            try:
                logger.info(
                    f"send_to_client (Attempt {attempt+1}/{SEND_TO_CLIENT_MAX_RETRIES+1}): Sending to {client_Actor_id} at {url}"
                )
                response = await asyncio.to_thread(_blocking_post_request)
                response.raise_for_status()  # Raises HTTPError for 4xx/5xx status
                response_data = response.json()
                client_text_response = response_data.get("text")
                encoded_audio_data = response_data.get("audio_data")
                logger.info(
                    f"send_to_client: Received response from {client_Actor_id}. Text: '{str(client_text_response)[:50]}...', Audio present: {bool(encoded_audio_data)}"
                )

                if encoded_audio_data and pygame.mixer.get_init():
                    # This part (decode, save, play) is also blocking
                    def _handle_audio():
                        """
                        Decodes base64-encoded audio data, saves it as a WAV file in a character-specific directory, and plays the audio using pygame.

                        The audio file is named using the client Actor ID and a unique identifier to avoid collisions.
                        """
                        sane_char_name = "".join(
                            c if c.isalnum() else "_"
                            for c in character.get("name", client_Actor_id)
                        )
                        audio_dir = os.path.join(CHARACTERS_AUDIO_PATH, sane_char_name)
                        os.makedirs(audio_dir, exist_ok=True)
                        audio_filename = f"{client_Actor_id}_{uuid.uuid4()}.wav"
                        audio_path = os.path.join(audio_dir, audio_filename)
                        try:
                            decoded_audio_data = base64.b64decode(encoded_audio_data)
                            with open(audio_path, "wb") as f:
                                f.write(decoded_audio_data)
                            logger.info(
                                f"Saved client audio for {client_Actor_id} to {audio_path}"
                            )
                            pygame.mixer.Sound(audio_path).play()
                            logger.info(
                                f"Playing audio for {client_Actor_id} from {audio_path}"
                            )
                        except Exception as e_audio:
                            logger.error(
                                f"Error handling audio for {client_Actor_id} (path: {audio_path}): {e_audio}",
                                exc_info=True,
                            )

                    await asyncio.to_thread(_handle_audio)

                self.db.update_client_status(
                    client_Actor_id, "Online_Responsive"
                )  # Blocking DB call
                return client_text_response

            except requests.exceptions.Timeout:
                logger.warning(
                    f"send_to_client (Attempt {attempt+1}): Timeout for {client_Actor_id} at {url}."
                )
                if attempt == SEND_TO_CLIENT_MAX_RETRIES:
                    self.db.update_client_status(
                        client_Actor_id, "Error_API"
                    )  # Blocking
            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"send_to_client (Attempt {attempt+1}): Connection error for {client_Actor_id} at {url}."
                )
                if attempt == SEND_TO_CLIENT_MAX_RETRIES:
                    self.db.update_client_status(
                        client_Actor_id, "Error_Unreachable"
                    )  # Blocking
            except requests.exceptions.RequestException as e:  # Includes HTTPError
                logger.warning(
                    f"send_to_client (Attempt {attempt+1}): Request error for {client_Actor_id} at {url}: {e}"
                )
                if attempt == SEND_TO_CLIENT_MAX_RETRIES:
                    self.db.update_client_status(
                        client_Actor_id, "Error_API"
                    )  # Blocking
            except (
                Exception
            ) as e:  # Catch other errors like JSONDecodeError from response.json()
                logger.error(
                    f"send_to_client (Attempt {attempt+1}): Unexpected error for {client_Actor_id} at {url}: {e}",
                    exc_info=True,
                )
                if attempt == SEND_TO_CLIENT_MAX_RETRIES:
                    self.db.update_client_status(
                        client_Actor_id, "Error_API"
                    )  # Blocking

            if attempt < SEND_TO_CLIENT_MAX_RETRIES:
                delay = SEND_TO_CLIENT_BASE_DELAY_SECONDS * (2**attempt)
                logger.info(
                    f"send_to_client: Waiting {delay}s before retry for {client_Actor_id}..."
                )
                await asyncio.sleep(delay)  # Use asyncio.sleep for async context

        logger.error(f"send_to_client for {client_Actor_id} failed after all retries.")
        return ""  # Return empty if all retries fail

    def deactivate_client_Actor(self, Actor_id: str):  # Blocking DB call
        """
        Mark the specified client as deactivated in the database.

        Parameters:
            Actor_id (str): The identifier of the client to deactivate.
        """
        self.db.update_client_status(Actor_id, "Deactivated")
        logger.info(f"Client {Actor_id} marked as Deactivated.")

    def __del__(self):
        """
        Ensures that periodic health checks are stopped when the ClientManager instance is destroyed.
        """
        logger.debug("ClientManager instance being deleted. Stopping health checks.")
        self.stop_periodic_health_checks()
