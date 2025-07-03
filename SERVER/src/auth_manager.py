import secrets
from datetime import datetime, timezone, timedelta
import logging
from typing import Any # For type hinting

logger = logging.getLogger("dreamweaver_server")

class AuthManager:
    def __init__(self):
        """
        Manages authentication-related functionalities, specifically handshake challenges.
        """
        self.active_challenges: dict[str, dict[str, Any]] = {}  # Actor_id -> {"challenge": str, "timestamp": datetime}
        self.CHALLENGE_EXPIRY_SECONDS: int = 60 * 5 # 5 minutes, was 60 in client_manager, making it longer as per server_api.py suggestion
        logger.info(f"AuthManager initialized. Challenge expiry set to {self.CHALLENGE_EXPIRY_SECONDS} seconds.")

    def generate_handshake_challenge(self, actor_id: str) -> str | None:
        """
        Generates a new handshake challenge for the given Actor_id, stores it with a timestamp,
        and returns the challenge string.
        """
        if not actor_id:
            logger.warning("generate_handshake_challenge called with empty actor_id.")
            return None

        challenge = secrets.token_urlsafe(32)
        self.active_challenges[actor_id] = {
            "challenge": challenge,
            "timestamp": datetime.now(timezone.utc),
        }
        logger.info(f"Generated handshake challenge for Actor_id: {actor_id}")
        return challenge

    def get_and_validate_challenge(self, actor_id: str) -> str | None:
        """
        Retrieves the stored challenge for an Actor_id if it exists and hasn't expired.
        Does NOT remove the challenge; removal should happen after successful validation of the response.
        """
        if not actor_id:
            logger.warning("get_and_validate_challenge called with empty actor_id.")
            return None

        challenge_data = self.active_challenges.get(actor_id)
        if not challenge_data:
            logger.warning(f"No active handshake challenge found for Actor_id: {actor_id}")
            return None

        issue_time = challenge_data["timestamp"]
        if datetime.now(timezone.utc) - issue_time > timedelta(
            seconds=self.CHALLENGE_EXPIRY_SECONDS
        ):
            logger.warning(
                f"Handshake challenge expired for Actor_id: {actor_id}. Issued at: {issue_time}. Clearing it."
            )
            # Clean up expired challenge proactively
            self.clear_challenge(actor_id)
            return None

        logger.debug(f"Validated active handshake challenge for Actor_id: {actor_id}.")
        return challenge_data["challenge"]

    def clear_challenge(self, actor_id: str) -> None:
        """
        Removes a challenge for an Actor_id, typically after successful use or expiry.
        """
        if not actor_id:
            logger.warning("clear_challenge called with empty actor_id.")
            return

        if actor_id in self.active_challenges:
            del self.active_challenges[actor_id]
            logger.info(f"Cleared handshake challenge for Actor_id: {actor_id}")
        else:
            logger.debug(f"clear_challenge: No active challenge to clear for Actor_id: {actor_id}")

    # Future methods could include:
    # - More sophisticated token management (e.g., JWT)
    # - Role-based access control logic
    # - API key management
    # - etc.
pass
