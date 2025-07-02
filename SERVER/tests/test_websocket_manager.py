import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import logging

# Add SERVER/src to sys.path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from websocket_manager import WebSocketConnectionManager, connection_manager as global_connection_manager

# Reset the global connection_manager before each test module if it's used directly by other modules
# For unit testing WebSocketConnectionManager class, we'll instantiate it.

@pytest.fixture
def manager():
    """Fixture to create a new WebSocketConnectionManager instance for each test."""
    # Reset the global one if it's being imported and potentially modified elsewhere,
    # though for class unit tests, instantiating directly is cleaner.
    # For now, let's assume we test a fresh instance.
    return WebSocketConnectionManager()

@pytest.fixture
def mock_websocket():
    """Fixture to create a mock WebSocket object."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws

@pytest.mark.asyncio
class TestWebSocketConnectionManager:

    async def test_init(self, manager: WebSocketConnectionManager):
        assert manager.active_connections == {}

    async def test_connect(self, manager: WebSocketConnectionManager, mock_websocket: AsyncMock):
        actor_id = "actor1"
        await manager.connect(mock_websocket, actor_id)

        mock_websocket.accept.assert_awaited_once()
        assert actor_id in manager.active_connections
        assert manager.active_connections[actor_id] == mock_websocket
        assert len(manager.active_connections) == 1

    def test_disconnect_existing(self, manager: WebSocketConnectionManager, mock_websocket: AsyncMock):
        actor_id = "actor1"
        # Manually add to simulate prior connection
        manager.active_connections[actor_id] = mock_websocket

        manager.disconnect(actor_id)
        assert actor_id not in manager.active_connections
        assert len(manager.active_connections) == 0

    def test_disconnect_non_existing(self, manager: WebSocketConnectionManager):
        # Ensure logger is patched to capture warning for non-existing disconnect
        logger = logging.getLogger("dreamweaver_server")
        with patch.object(logger, 'warning') as mock_log_warning:
            manager.disconnect("non_actor")
            assert "non_actor" not in manager.active_connections
            mock_log_warning.assert_called_once_with(
                "Attempted to disconnect Actor_id: non_actor, but no active WebSocket connection found."
            )


    async def test_send_personal_message_success(self, manager: WebSocketConnectionManager, mock_websocket: AsyncMock):
        actor_id = "actor1"
        message = {"type": "greeting", "text": "hello"}
        manager.active_connections[actor_id] = mock_websocket # Simulate connected client

        result = await manager.send_personal_message(message, actor_id)

        assert result is True
        mock_websocket.send_json.assert_awaited_once_with(message)

    async def test_send_personal_message_non_existing_actor(self, manager: WebSocketConnectionManager):
        message = {"type": "info", "data": "test"}
        logger = logging.getLogger("dreamweaver_server")
        with patch.object(logger, 'warning') as mock_log_warning:
            result = await manager.send_personal_message(message, "unknown_actor")
            assert result is False
            mock_log_warning.assert_called_once() # Check that a warning was logged

    async def test_send_personal_message_send_failure_disconnects(self, manager: WebSocketConnectionManager, mock_websocket: AsyncMock):
        actor_id = "actor_fail"
        message = {"type": "error_test"}
        manager.active_connections[actor_id] = mock_websocket

        mock_websocket.send_json.side_effect = Exception("Connection broken")

        logger = logging.getLogger("dreamweaver_server")
        with patch.object(logger, 'error') as mock_log_error:
            result = await manager.send_personal_message(message, actor_id)
            assert result is False
            mock_log_error.assert_called_once() # Check error was logged

        # Verify client was disconnected
        assert actor_id not in manager.active_connections

    async def test_broadcast_success(self, manager: WebSocketConnectionManager):
        ws1 = AsyncMock(spec=["send_json"])
        ws2 = AsyncMock(spec=["send_json"])
        manager.active_connections["actor1"] = ws1
        manager.active_connections["actor2"] = ws2
        message = {"type": "broadcast", "content": "all"}

        await manager.broadcast(message)

        ws1.send_json.assert_awaited_once_with(message)
        ws2.send_json.assert_awaited_once_with(message)

    async def test_broadcast_partial_failure(self, manager: WebSocketConnectionManager):
        ws_ok = AsyncMock(spec=["send_json"])
        ws_fail = AsyncMock(spec=["send_json"])
        ws_fail.send_json.side_effect = Exception("Send failed on ws_fail")

        manager.active_connections["actor_ok"] = ws_ok
        manager.active_connections["actor_fail"] = ws_fail
        message = {"type": "partial_broadcast"}

        logger = logging.getLogger("dreamweaver_server")
        with patch.object(logger, 'error') as mock_log_error:
            await manager.broadcast(message)

        ws_ok.send_json.assert_awaited_once_with(message)
        ws_fail.send_json.assert_awaited_once_with(message) # Attempted send

        mock_log_error.assert_called_once() # Error for ws_fail
        assert "actor_ok" in manager.active_connections # OK client remains
        assert "actor_fail" not in manager.active_connections # Failed client disconnected

    async def test_broadcast_empty(self, manager: WebSocketConnectionManager):
        message = {"type": "empty_broadcast"}
        # No connections active
        await manager.broadcast(message)
        # Should not raise any error, just do nothing effectively

    def test_get_active_clients(self, manager: WebSocketConnectionManager):
        assert manager.get_active_clients() == []

        manager.active_connections["actor1"] = AsyncMock()
        manager.active_connections["actor2"] = AsyncMock()

        active_clients = manager.get_active_clients()
        assert len(active_clients) == 2
        assert "actor1" in active_clients
        assert "actor2" in active_clients

# To run these tests, use pytest from the terminal in the SERVER directory (or project root)
# Example: pytest tests/test_websocket_manager.py
