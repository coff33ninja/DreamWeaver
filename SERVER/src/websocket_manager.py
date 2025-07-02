import asyncio
from fastapi import WebSocket # type: ignore # Using type: ignore for fastapi.WebSocket if linters struggle
from typing import Dict, List
import logging
import json

logger = logging.getLogger("dreamweaver_server")

class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        logger.info("WebSocketConnectionManager initialized.")

    async def connect(self, websocket: WebSocket, actor_id: str):
        """
        Accepts a new WebSocket connection and stores it.
        """
        await websocket.accept()
        self.active_connections[actor_id] = websocket
        logger.info(f"WebSocket connection established for Actor_id: {actor_id}. Total connections: {len(self.active_connections)}")

    def disconnect(self, actor_id: str):
        """
        Removes a WebSocket connection.
        """
        if actor_id in self.active_connections:
            # WebSocket object itself might be closed by FastAPI already,
            # this is mainly for cleaning up our tracking.
            del self.active_connections[actor_id]
            logger.info(f"WebSocket connection closed for Actor_id: {actor_id}. Total connections: {len(self.active_connections)}")
        else:
            logger.warning(f"Attempted to disconnect Actor_id: {actor_id}, but no active WebSocket connection found.")

    async def send_personal_message(self, message: dict, actor_id: str) -> bool:
        """
        Sends a JSON message to a specific connected client.
        Returns True if message was sent, False otherwise.
        """
        websocket = self.active_connections.get(actor_id)
        if websocket:
            try:
                await websocket.send_json(message)
                logger.info(f"Sent WebSocket message to Actor_id {actor_id}: {str(message)[:100]}...")
                return True
            except Exception as e: # Handles if websocket is already closed unexpectedly etc.
                logger.error(f"Error sending WebSocket message to Actor_id {actor_id}: {e}", exc_info=True)
                # If sending fails, the connection is likely broken. Remove it.
                self.disconnect(actor_id)
                return False
        else:
            logger.warning(f"No active WebSocket connection found for Actor_id {actor_id} to send message: {str(message)[:100]}...")
            return False

    async def broadcast(self, message: dict): # Not used initially, but good to have
        """
        Broadcasts a JSON message to all connected clients.
        """
        logger.info(f"Broadcasting WebSocket message to {len(self.active_connections)} clients: {str(message)[:100]}...")
        disconnected_clients: List[str] = []
        for actor_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to Actor_id {actor_id}: {e}", exc_info=True)
                disconnected_clients.append(actor_id)

        # Clean up connections that failed during broadcast
        for actor_id in disconnected_clients:
            self.disconnect(actor_id)

    def get_active_clients(self) -> List[str]:
        """
        Returns a list of Actor_ids for all currently active WebSocket connections.
        """
        return list(self.active_connections.keys())

# Global instance (or manage via FastAPI dependencies if preferred for more complex scenarios)
# For now, a global instance is simpler for Gradio to access if needed.
connection_manager = WebSocketConnectionManager()
