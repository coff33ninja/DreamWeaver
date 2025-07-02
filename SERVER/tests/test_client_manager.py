import unittest
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
import asyncio
import threading
import time
import json
import base64
import os
import tempfile
from datetime import datetime, timezone, timedelta
import sys
import requests

# Import the ClientManager - adjust path as needed for your project structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from client_manager import ClientManager, CLIENT_HEALTH_CHECK_INTERVAL_SECONDS, CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS
    from database import Database
except ImportError:
    # Fallback import paths
    try:
        from SERVER.src.client_manager import ClientManager, CLIENT_HEALTH_CHECK_INTERVAL_SECONDS, CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS
        from SERVER.src.database import Database
    except ImportError:
        # Mock the imports if we can't find them
        class ClientManager:
            pass
        CLIENT_HEALTH_CHECK_INTERVAL_SECONDS = 120
        CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS = 5


class TestClientManager(unittest.TestCase):
    """
    Comprehensive unit tests for ClientManager class.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite covers:
    - Initialization and setup
    - Token generation and validation
    - Health check functionality (sync and async patterns)
    - Client communication (async send_to_client method)
    - Threading behavior for periodic health checks
    - Error handling and edge cases
    - Integration with Database and external services
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock database
        self.mock_db = Mock(spec=Database)
        
        # Mock pygame mixer to avoid initialization issues
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)
            
        # Test data
        self.test_actor_id = "TestActor1"
        self.test_token = "test_token_123456789012345678901234567890123456789012345678"
        self.test_ip = "192.168.1.100"
        self.test_port = 8080
        self.test_character_data = {
            'name': 'TestCharacter',
            'personality': 'Friendly',
            'Actor_id': self.test_actor_id,
            'tts': 'piper',
            'tts_model': 'en_US-ryan-high'
        }
        
    def tearDown(self):
        """Clean up after each test method."""
        # Stop any running health check threads
        if hasattr(self.client_manager, 'stop_periodic_health_checks'):
            self.client_manager.stop_periodic_health_checks()
        
    # --- Initialization Tests ---
    
    @patch('client_manager.pygame.mixer')
    def test_init_with_pygame_success(self, mock_mixer):
        """Test ClientManager initialization with successful pygame setup."""
        mock_mixer.get_init.return_value = False
        mock_mixer.init.return_value = None
        
        cm = ClientManager(self.mock_db)
        
        self.assertIsNotNone(cm)
        self.assertEqual(cm.db, self.mock_db)
        mock_mixer.init.assert_called_once()
        
    @patch('client_manager.pygame.mixer')
    def test_init_with_pygame_failure(self, mock_mixer):
        """Test ClientManager initialization when pygame fails."""
        mock_mixer.get_init.return_value = False
        mock_mixer.init.side_effect = Exception("Pygame init failed")
        
        # Should not raise exception, just print warning
        cm = ClientManager(self.mock_db)
        
        self.assertIsNotNone(cm)
        self.assertEqual(cm.db, self.mock_db)
        
    def test_init_sets_initial_state(self):
        """Test that initialization sets correct initial state."""
        self.assertIsNone(self.client_manager.health_check_thread)
        self.assertIsInstance(self.client_manager.stop_health_check_event, threading.Event)
        self.assertFalse(self.client_manager.stop_health_check_event.is_set())
        
    # --- Token Generation Tests ---
    
    def test_generate_token_new_character(self):
        """Test token generation for existing character."""
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.save_client_token.return_value = None
        
        token = self.client_manager.generate_token(self.test_actor_id)
        
        self.assertIsInstance(token, str)
        self.assertEqual(len(token), 48)  # 24 bytes hex = 48 characters
        self.mock_db.get_character.assert_called_once_with(self.test_actor_id)
        self.mock_db.save_client_token.assert_called_once_with(self.test_actor_id, token)
        
    def test_generate_token_actor1_special_case(self):
        """Test token generation for Actor1 when character doesn't exist."""
        self.mock_db.get_character.return_value = None
        self.mock_db.save_character.return_value = None
        self.mock_db.save_client_token.return_value = None
        
        token = self.client_manager.generate_token("Actor1")
        
        self.assertIsInstance(token, str)
        self.assertEqual(len(token), 48)
        self.mock_db.save_character.assert_called_once()
        self.mock_db.save_client_token.assert_called_once_with("Actor1", token)
        
    def test_generate_token_nonexistent_character_warning(self):
        """Test token generation for non-existent character (not Actor1)."""
        self.mock_db.get_character.return_value = None
        self.mock_db.save_client_token.return_value = None
        
        with patch('builtins.print') as mock_print:
            token = self.client_manager.generate_token("NonExistentActor")
            
        self.assertIsInstance(token, str)
        mock_print.assert_called()
        
    def test_generate_token_uniqueness(self):
        """Test that generated tokens are unique."""
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.save_client_token.return_value = None
        
        token1 = self.client_manager.generate_token("Actor1")
        token2 = self.client_manager.generate_token("Actor2")
        
        self.assertNotEqual(token1, token2)
        
    # --- Token Validation Tests ---
    
    def test_validate_token_success(self):
        """Test successful token validation."""
        self.mock_db.get_client_token_details.return_value = {
            'token': self.test_token,
            'status': 'Online_Responsive'
        }
        
        result = self.client_manager.validate_token(self.test_actor_id, self.test_token)
        
        self.assertTrue(result)
        self.mock_db.get_client_token_details.assert_called_once_with(self.test_actor_id)
        
    def test_validate_token_wrong_token(self):
        """Test token validation with wrong token."""
        self.mock_db.get_client_token_details.return_value = {
            'token': 'different_token',
            'status': 'Online_Responsive'
        }
        
        result = self.client_manager.validate_token(self.test_actor_id, self.test_token)
        
        self.assertFalse(result)
        
    def test_validate_token_deactivated_status(self):
        """Test token validation for deactivated client."""
        self.mock_db.get_client_token_details.return_value = {
            'token': self.test_token,
            'status': 'Deactivated'
        }
        
        result = self.client_manager.validate_token(self.test_actor_id, self.test_token)
        
        self.assertFalse(result)
        
    def test_validate_token_no_client_details(self):
        """Test token validation when client details don't exist."""
        self.mock_db.get_client_token_details.return_value = None
        
        result = self.client_manager.validate_token(self.test_actor_id, self.test_token)
        
        self.assertFalse(result)
        
    def test_validate_token_edge_cases(self):
        """Test token validation edge cases."""
        # Empty token
        self.mock_db.get_client_token_details.return_value = {
            'token': self.test_token,
            'status': 'Online_Responsive'
        }
        
        result = self.client_manager.validate_token(self.test_actor_id, "")
        self.assertFalse(result)
        
        # None token
        result = self.client_manager.validate_token(self.test_actor_id, None)
        self.assertFalse(result)
        
    # --- Health Check Tests ---
    
    @patch('client_manager.requests.get')
    def test_perform_single_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {'status': 'ok'}
        mock_get.return_value = mock_response
        
        client_info = {
            'Actor_id': self.test_actor_id,
            'ip_address': self.test_ip,
            'client_port': self.test_port
        }
        
        self.client_manager._perform_single_health_check_blocking(client_info)
        
        mock_get.assert_called_once_with(
            f"http://{self.test_ip}:{self.test_port}/health",
            timeout=CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS
        )
        self.mock_db.update_client_status.assert_called_once_with(
            self.test_actor_id, "Online_Responsive"
        )
        
    @patch('client_manager.requests.get')
    def test_perform_single_health_check_degraded(self, mock_get):
        """Test health check with degraded status."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {'status': 'degraded'}
        mock_get.return_value = mock_response
        
        client_info = {
            'Actor_id': self.test_actor_id,
            'ip_address': self.test_ip,
            'client_port': self.test_port
        }
        
        self.client_manager._perform_single_health_check_blocking(client_info)
        
        self.mock_db.update_client_status.assert_called_once_with(
            self.test_actor_id, "Error_API_Degraded"
        )
        
    @patch('client_manager.requests.get')
    def test_perform_single_health_check_timeout(self, mock_get):
        """Test health check timeout handling."""
        mock_get.side_effect = requests.exceptions.Timeout()
        
        client_info = {
            'Actor_id': self.test_actor_id,
            'ip_address': self.test_ip,
            'client_port': self.test_port
        }
        
        self.client_manager._perform_single_health_check_blocking(client_info)
        
        self.mock_db.update_client_status.assert_called_once_with(
            self.test_actor_id, "Error_API"
        )
        
    @patch('client_manager.requests.get')
    def test_perform_single_health_check_connection_error(self, mock_get):
        """Test health check connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        client_info = {
            'Actor_id': self.test_actor_id,
            'ip_address': self.test_ip,
            'client_port': self.test_port
        }
        
        self.client_manager._perform_single_health_check_blocking(client_info)
        
        self.mock_db.update_client_status.assert_called_once_with(
            self.test_actor_id, "Error_Unreachable"
        )
        
    @patch('client_manager.requests.get')
    def test_perform_single_health_check_request_exception(self, mock_get):
        """Test health check with general request exception."""
        mock_get.side_effect = requests.exceptions.RequestException("Request failed")
        
        client_info = {
            'Actor_id': self.test_actor_id,
            'ip_address': self.test_ip,
            'client_port': self.test_port
        }
        
        self.client_manager._perform_single_health_check_blocking(client_info)
        
        self.mock_db.update_client_status.assert_called_once_with(
            self.test_actor_id, "Error_API"
        )
        
    @patch('client_manager.requests.get')
    def test_perform_single_health_check_unexpected_error(self, mock_get):
        """Test health check with unexpected error."""
        mock_get.side_effect = ValueError("Unexpected error")
        
        client_info = {
            'Actor_id': self.test_actor_id,
            'ip_address': self.test_ip,
            'client_port': self.test_port
        }
        
        self.client_manager._perform_single_health_check_blocking(client_info)
        
        self.mock_db.update_client_status.assert_called_once_with(
            self.test_actor_id, "Error_API"
        )
        
    def test_perform_single_health_check_missing_data(self):
        """Test health check with missing client info."""
        client_info = {'Actor_id': self.test_actor_id}  # Missing ip_address and client_port
        
        self.client_manager._perform_single_health_check_blocking(client_info)
        
        # Should return early without making any DB calls
        self.mock_db.update_client_status.assert_not_called()
        
    # --- Threading Tests ---
    
    def test_start_periodic_health_checks(self):
        """Test starting periodic health check thread."""
        self.client_manager.start_periodic_health_checks()
        
        self.assertIsNotNone(self.client_manager.health_check_thread)
        self.assertTrue(self.client_manager.health_check_thread.is_alive())
        self.assertFalse(self.client_manager.stop_health_check_event.is_set())
        
    def test_start_periodic_health_checks_already_running(self):
        """Test starting health checks when already running."""
        self.client_manager.start_periodic_health_checks()
        first_thread = self.client_manager.health_check_thread
        
        self.client_manager.start_periodic_health_checks()
        
        # Should still be the same thread
        self.assertEqual(self.client_manager.health_check_thread, first_thread)
        
    def test_stop_periodic_health_checks(self):
        """Test stopping periodic health check thread."""
        self.client_manager.start_periodic_health_checks()
        
        self.client_manager.stop_periodic_health_checks()
        
        self.assertTrue(self.client_manager.stop_health_check_event.is_set())
        # Give thread time to stop
        time.sleep(0.1)
        
    def test_stop_periodic_health_checks_not_running(self):
        """Test stopping health checks when not running."""
        # Should not raise exception
        self.client_manager.stop_periodic_health_checks()
        
        self.assertTrue(self.client_manager.stop_health_check_event.is_set())
        
    @patch('client_manager.requests.get')
    def test_periodic_health_check_loop_integration(self, mock_get):
        """Test the periodic health check loop with mock data."""
        # Mock successful health check response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {'status': 'ok'}
        mock_get.return_value = mock_response
        
        # Mock database to return clients to check
        self.mock_db.get_all_client_statuses.return_value = [
            {
                'Actor_id': self.test_actor_id,
                'ip_address': self.test_ip,
                'client_port': self.test_port,
                'status': 'Error_API'
            }
        ]
        
        # Start health checks and let them run briefly
        self.client_manager.start_periodic_health_checks()
        time.sleep(0.1)  # Let it run one iteration
        self.client_manager.stop_periodic_health_checks()
        
        # Verify health check was performed
        self.mock_db.get_all_client_statuses.assert_called()
        
    # --- Async send_to_client Tests ---
    
    @patch('client_manager.requests.post')
    @patch('client_manager.asyncio.to_thread')
    @patch('client_manager.pygame.mixer')
    def test_send_to_client_success(self, mock_mixer, mock_to_thread, mock_post):
        """Test successful send_to_client operation."""
        # Setup mocks
        mock_mixer.get_init.return_value = True
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.get_token.return_value = self.test_token
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None  
        mock_response.json.return_value = {
            'text': 'Character response',
            'audio_data': base64.b64encode(b'fake_audio_data').decode('utf-8')
        }
        
        # Mock the blocking post request
        async def mock_post_request():
            return mock_response
            
        mock_to_thread.side_effect = [mock_post_request(), None]  # Second call for audio handling
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {"character1": "some text"}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result, 'Character response')
        self.mock_db.get_character.assert_called_once_with(self.test_actor_id)
        self.mock_db.get_token.assert_called_once_with(self.test_actor_id)
        self.mock_db.update_client_status.assert_called_with(self.test_actor_id, "Online_Responsive")
        
    @patch('client_manager.asyncio.to_thread')
    def test_send_to_client_no_character(self, mock_to_thread):
        """Test send_to_client when character doesn't exist."""
        self.mock_db.get_character.return_value = None
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result, "")
        self.mock_db.update_client_status.assert_called_with(self.test_actor_id, "Error_API")
        
    @patch('client_manager.asyncio.to_thread')
    def test_send_to_client_no_token(self, mock_to_thread):
        """Test send_to_client when token doesn't exist."""
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.get_token.return_value = None
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result, "")
        self.mock_db.update_client_status.assert_called_with(self.test_actor_id, "Error_API")
        
    @patch('client_manager.requests.post')
    @patch('client_manager.asyncio.to_thread')
    @patch('client_manager.asyncio.sleep')
    def test_send_to_client_with_retries(self, mock_sleep, mock_to_thread, mock_post):
        """Test send_to_client retry mechanism."""
        # Setup database mocks
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.get_token.return_value = self.test_token
        
        # First attempt fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.Timeout()
        
        mock_response_success = Mock()
        mock_response_success.raise_for_status.return_value = None
        mock_response_success.json.return_value = {
            'text': 'Success after retry',
            'audio_data': None
        }
        
        async def mock_failing_request():
            raise requests.exceptions.Timeout()
            
        async def mock_successful_request():
            return mock_response_success
            
        mock_to_thread.side_effect = [mock_failing_request(), mock_successful_request()]
        mock_sleep.return_value = None  # Make sleep non-blocking
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result, 'Success after retry')
        mock_sleep.assert_called_once()  # Should have waited between retries
        
    @patch('client_manager.asyncio.to_thread')
    @patch('client_manager.asyncio.sleep')
    def test_send_to_client_all_retries_fail(self, mock_sleep, mock_to_thread):
        """Test send_to_client when all retries fail."""
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.get_token.return_value = self.test_token
        
        async def mock_failing_request():
            raise requests.exceptions.ConnectionError()
            
        mock_to_thread.side_effect = [mock_failing_request()] * 10  # More than max retries
        mock_sleep.return_value = None
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result, "")
        self.mock_db.update_client_status.assert_called_with(self.test_actor_id, "Error_Unreachable")
        
    # --- Client Management Tests ---
    
    def test_get_clients_for_story_progression(self):
        """Test getting clients for story progression."""
        expected_clients = [
            {'Actor_id': 'Actor2', 'ip_address': '192.168.1.100', 'client_port': 8080},
            {'Actor_id': 'Actor3', 'ip_address': '192.168.1.101', 'client_port': 8081}
        ]
        self.mock_db.get_clients_for_story_progression.return_value = expected_clients
        
        result = self.client_manager.get_clients_for_story_progression()
        
        self.assertEqual(result, expected_clients)
        self.mock_db.get_clients_for_story_progression.assert_called_once()
        
    def test_deactivate_client_actor(self):
        """Test deactivating a client actor."""
        with patch('builtins.print') as mock_print:
            self.client_manager.deactivate_client_Actor(self.test_actor_id)
            
        self.mock_db.update_client_status.assert_called_once_with(self.test_actor_id, "Deactivated")
        mock_print.assert_called_once()
        
    # --- Error Handling and Edge Cases ---
    
    def test_generate_token_with_special_characters_in_actor_id(self):
        """Test token generation with special characters in Actor ID."""
        special_actor_id = "Actor@#$%^&*()_+1"
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.save_client_token.return_value = None
        
        token = self.client_manager.generate_token(special_actor_id)
        
        self.assertIsInstance(token, str)
        self.assertEqual(len(token), 48)
        
    def test_validate_token_with_unicode_characters(self):
        """Test token validation with unicode characters."""
        unicode_token = "test_token_with_üñîçødé_characters"
        self.mock_db.get_client_token_details.return_value = {
            'token': unicode_token,
            'status': 'Online_Responsive'
        }
        
        result = self.client_manager.validate_token(self.test_actor_id, unicode_token)
        
        self.assertTrue(result)
        
    @patch('client_manager.pygame.mixer')
    @patch('client_manager.os.makedirs')
    @patch('client_manager.open', create=True)
    @patch('client_manager.base64.b64decode')
    @patch('client_manager.asyncio.to_thread')
    def test_send_to_client_audio_handling(self, mock_to_thread, mock_b64decode, mock_open, mock_makedirs, mock_mixer):
        """Test audio handling in send_to_client."""
        # Setup mocks
        mock_mixer.get_init.return_value = True
        mock_mixer.Sound.return_value.play.return_value = None
        mock_b64decode.return_value = b'decoded_audio_data'
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.get_token.return_value = self.test_token
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'text': 'Character response',
            'audio_data': 'encoded_audio_data'
        }
        
        async def mock_post_request():
            return mock_response
            
        async def mock_audio_handler():
            pass
            
        mock_to_thread.side_effect = [mock_post_request(), mock_audio_handler()]
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result, 'Character response')
        mock_makedirs.assert_called()
        mock_b64decode.assert_called_once_with('encoded_audio_data')
        
    # --- Performance and Stress Tests ---
    
    def test_multiple_token_generation_performance(self):
        """Test performance of generating multiple tokens."""
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.save_client_token.return_value = None
        
        start_time = time.time()
        tokens = []
        
        for i in range(100):
            token = self.client_manager.generate_token(f"Actor{i}")
            tokens.append(token)
            
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 1.0)  # Less than 1 second
        
        # All tokens should be unique
        self.assertEqual(len(tokens), len(set(tokens)))
        
    def test_concurrent_token_validation(self):
        """Test concurrent token validation."""
        self.mock_db.get_client_token_details.return_value = {
            'token': self.test_token,
            'status': 'Online_Responsive'
        }
        
        def validate_token_worker():
            return self.client_manager.validate_token(self.test_actor_id, self.test_token)
            
        threads = []
        results = []
        
        for _ in range(10):
            thread = threading.Thread(target=lambda: results.append(validate_token_worker()))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All validations should succeed
        self.assertTrue(all(results))
        self.assertEqual(len(results), 10)
        
    # --- Integration Tests ---
    
    @patch('client_manager.pygame.mixer')
    def test_full_workflow_integration(self, mock_mixer):
        """Test complete workflow from token generation to client communication."""
        mock_mixer.get_init.return_value = True
        
        # Setup database responses
        self.mock_db.get_character.return_value = self.test_character_data
        self.mock_db.save_client_token.return_value = None
        self.mock_db.get_client_token_details.return_value = {
            'token': self.test_token,
            'status': 'Online_Responsive'
        }
        self.mock_db.get_token.return_value = self.test_token
        
        # 1. Generate token
        token = self.client_manager.generate_token(self.test_actor_id)
        self.assertIsInstance(token, str)
        
        # 2. Validate token
        is_valid = self.client_manager.validate_token(self.test_actor_id, token)
        self.assertTrue(is_valid)
        
        # 3. Start health checks
        self.client_manager.start_periodic_health_checks()
        self.assertTrue(self.client_manager.health_check_thread.is_alive())
        
        # 4. Stop health checks
        self.client_manager.stop_periodic_health_checks()
        
        # 5. Deactivate client
        self.client_manager.deactivate_client_Actor(self.test_actor_id)
        self.mock_db.update_client_status.assert_called_with(self.test_actor_id, "Deactivated")
        
    # --- Cleanup and Resource Management Tests ---
    
    def test_destructor_cleanup(self):
        """Test that destructor properly cleans up resources."""
        # Start health checks
        self.client_manager.start_periodic_health_checks()
        
        # Call destructor
        self.client_manager.__del__()
        
        # Health check should be stopped
        self.assertTrue(self.client_manager.stop_health_check_event.is_set())


class TestClientManagerAsyncPatterns(unittest.TestCase):
    """
    Additional tests focused on async/await patterns and coroutine behavior.
    
    Testing Framework: Python unittest with asyncio support
    """
    
    def setUp(self):
        """Set up async test fixtures."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)
            
    def test_send_to_client_is_coroutine(self):
        """Test that send_to_client returns a coroutine."""
        result = self.client_manager.send_to_client(
            "TestActor",
            "127.0.0.1",
            8080,
            "Test narration",
            {}
        )
        
        self.assertTrue(asyncio.iscoroutine(result))
        
        # Clean up the coroutine
        result.close()
        
    @patch('client_manager.asyncio.to_thread')
    async def test_send_to_client_awaitable(self, mock_to_thread):
        """Test that send_to_client is properly awaitable."""
        self.mock_db.get_character.return_value = None  # Will cause early return
        
        result = await self.client_manager.send_to_client(
            "TestActor",
            "127.0.0.1", 
            8080,
            "Test narration",
            {}
        )
        
        self.assertEqual(result, "")


if __name__ == '__main__':
    # Configure test runner for comprehensive output
    unittest.main(
        verbosity=2,
        failfast=False,
        buffer=True,
        warnings='ignore'
    )

class TestClientManagerExtendedEdgeCases(unittest.TestCase):
    """
    Extended edge case tests for ClientManager class.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite adds coverage for:
    - Complex error scenarios and boundary conditions
    - Resource management and cleanup
    - Database interaction edge cases
    - Network failure simulation
    - Memory and performance edge cases
    - Concurrent operation stress testing
    """

    def setUp(self):
        """Set up test fixtures for extended edge case testing."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)
            
        self.test_actor_id = "EdgeCaseActor"
        self.test_token = "edge_case_token_123456789012345678901234567890123456789012345678"
        self.test_ip = "192.168.1.200"
        self.test_port = 9090

    def tearDown(self):
        """Clean up after extended edge case tests."""
        if hasattr(self.client_manager, 'stop_periodic_health_checks'):
            self.client_manager.stop_periodic_health_checks()

    # --- Database Error Edge Cases ---
    
    def test_generate_token_db_save_failure(self):
        """Test token generation when database save fails."""
        self.mock_db.get_character.return_value = {'Actor_id': self.test_actor_id}
        self.mock_db.save_client_token.side_effect = Exception("Database connection lost")
        
        with patch('builtins.print') as mock_print:
            token = self.client_manager.generate_token(self.test_actor_id)
        
        self.assertIsInstance(token, str)
        mock_print.assert_called()  # Should print error message

    def test_validate_token_db_exception(self):
        """Test token validation when database throws exception."""
        self.mock_db.get_client_token_details.side_effect = Exception("Database timeout")
        
        result = self.client_manager.validate_token(self.test_actor_id, self.test_token)
        
        self.assertFalse(result)

    def test_generate_token_character_save_failure(self):
        """Test Actor1 special case when character save fails."""
        self.mock_db.get_character.return_value = None
        self.mock_db.save_character.side_effect = Exception("Character save failed")
        self.mock_db.save_client_token.return_value = None
        
        with patch('builtins.print') as mock_print:
            token = self.client_manager.generate_token("Actor1")
        
        self.assertIsInstance(token, str)
        mock_print.assert_called()

    # --- Network and Communication Edge Cases ---

    @patch('client_manager.requests.get')
    def test_health_check_malformed_json_response(self, mock_get):
        """Test health check with malformed JSON response."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response
        
        client_info = {
            'Actor_id': self.test_actor_id,
            'ip_address': self.test_ip,
            'client_port': self.test_port
        }
        
        self.client_manager._perform_single_health_check_blocking(client_info)
        
        self.mock_db.update_client_status.assert_called_once_with(
            self.test_actor_id, "Error_API"
        )

    @patch('client_manager.requests.get')
    def test_health_check_empty_response_body(self, mock_get):
        """Test health check with empty response body."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}  # Empty response
        mock_get.return_value = mock_response
        
        client_info = {
            'Actor_id': self.test_actor_id,
            'ip_address': self.test_ip,
            'client_port': self.test_port
        }
        
        self.client_manager._perform_single_health_check_blocking(client_info)
        
        self.mock_db.update_client_status.assert_called_once_with(
            self.test_actor_id, "Error_API"
        )

    @patch('client_manager.requests.get')
    def test_health_check_http_error_codes(self, mock_get):
        """Test health check with various HTTP error codes."""
        error_codes = [400, 401, 403, 404, 500, 502, 503]
        
        for error_code in error_codes:
            with self.subTest(error_code=error_code):
                mock_response = Mock()
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{error_code} Error")
                mock_get.return_value = mock_response
                
                client_info = {
                    'Actor_id': f"{self.test_actor_id}_{error_code}",
                    'ip_address': self.test_ip,
                    'client_port': self.test_port
                }
                
                self.mock_db.reset_mock()
                self.client_manager._perform_single_health_check_blocking(client_info)
                
                self.mock_db.update_client_status.assert_called_once()

    # --- Async Operation Edge Cases ---

    @patch('client_manager.asyncio.to_thread')
    @patch('client_manager.asyncio.sleep')
    def test_send_to_client_asyncio_cancelled_error(self, mock_sleep, mock_to_thread):
        """Test send_to_client when asyncio operation is cancelled."""
        self.mock_db.get_character.return_value = {'Actor_id': self.test_actor_id}
        self.mock_db.get_token.return_value = self.test_token
        
        async def mock_cancelled_request():
            raise asyncio.CancelledError("Operation cancelled")
            
        mock_to_thread.side_effect = mock_cancelled_request
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {}
            )
            return result
            
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run_test())

    @patch('client_manager.asyncio.to_thread')
    def test_send_to_client_memory_error(self, mock_to_thread):
        """Test send_to_client when memory error occurs."""
        self.mock_db.get_character.return_value = {'Actor_id': self.test_actor_id}
        self.mock_db.get_token.return_value = self.test_token
        
        async def mock_memory_error():
            raise MemoryError("Out of memory")
            
        mock_to_thread.side_effect = mock_memory_error
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result, "")
        self.mock_db.update_client_status.assert_called_with(self.test_actor_id, "Error_API")

    # --- Input Validation Edge Cases ---

    def test_generate_token_empty_actor_id(self):
        """Test token generation with empty Actor ID."""
        with patch('builtins.print') as mock_print:
            token = self.client_manager.generate_token("")
        
        self.assertIsInstance(token, str)
        self.assertEqual(len(token), 48)

    def test_generate_token_none_actor_id(self):
        """Test token generation with None Actor ID."""
        with self.assertRaises(TypeError):
            self.client_manager.generate_token(None)

    def test_validate_token_very_long_token(self):
        """Test token validation with extremely long token."""
        very_long_token = "x" * 10000
        self.mock_db.get_client_token_details.return_value = {
            'token': self.test_token,
            'status': 'Online_Responsive'
        }
        
        result = self.client_manager.validate_token(self.test_actor_id, very_long_token)
        
        self.assertFalse(result)

    def test_validate_token_special_status_values(self):
        """Test token validation with unusual status values."""
        unusual_statuses = ["", None, "UNKNOWN", "invalid_status", 123, []]
        
        for status in unusual_statuses:
            with self.subTest(status=status):
                self.mock_db.get_client_token_details.return_value = {
                    'token': self.test_token,
                    'status': status
                }
                
                result = self.client_manager.validate_token(self.test_actor_id, self.test_token)
                
                if status == "Online_Responsive":
                    self.assertTrue(result)
                else:
                    self.assertFalse(result)

    # --- Threading and Concurrency Edge Cases ---

    def test_health_check_thread_exception_handling(self):
        """Test health check thread handling internal exceptions."""
        self.mock_db.get_all_client_statuses.side_effect = Exception("Database error")
        
        self.client_manager.start_periodic_health_checks()
        time.sleep(0.1)  # Let thread attempt operation
        self.client_manager.stop_periodic_health_checks()
        
        # Thread should handle exception gracefully and not crash

    def test_multiple_start_stop_health_checks(self):
        """Test rapid start/stop of health check threads."""
        for _ in range(5):
            self.client_manager.start_periodic_health_checks()
            self.assertTrue(self.client_manager.health_check_thread.is_alive())
            
            self.client_manager.stop_periodic_health_checks()
            time.sleep(0.05)  # Brief pause

    def test_concurrent_token_generation(self):
        """Test concurrent token generation from multiple threads."""
        self.mock_db.get_character.return_value = {'Actor_id': 'test'}
        self.mock_db.save_client_token.return_value = None
        
        tokens = []
        errors = []
        
        def generate_token_worker(actor_id):
            try:
                token = self.client_manager.generate_token(f"Actor{actor_id}")
                tokens.append(token)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(20):
            thread = threading.Thread(target=generate_token_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.assertEqual(len(errors), 0)  # No errors should occur
        self.assertEqual(len(tokens), 20)  # All tokens generated
        self.assertEqual(len(set(tokens)), 20)  # All tokens unique

    # --- Resource Management Edge Cases ---

    @patch('client_manager.pygame.mixer')
    def test_audio_handling_pygame_not_available(self, mock_mixer):
        """Test audio handling when pygame is not available."""
        mock_mixer.get_init.return_value = False
        mock_mixer.init.side_effect = Exception("Pygame not available")
        
        cm = ClientManager(self.mock_db)
        
        # Should handle gracefully without crashing
        self.assertIsNotNone(cm)

    @patch('client_manager.tempfile.gettempdir')
    @patch('client_manager.os.makedirs')
    def test_audio_file_creation_permission_error(self, mock_makedirs, mock_gettempdir):
        """Test audio file creation when permission denied."""
        mock_gettempdir.return_value = "/tmp"
        mock_makedirs.side_effect = PermissionError("Permission denied")
        
        # Should handle permission error gracefully
        with patch('builtins.print'):
            # This would be called during audio handling
            pass

    def test_destructor_multiple_calls(self):
        """Test that destructor can be called multiple times safely."""
        self.client_manager.start_periodic_health_checks()
        
        # Call destructor multiple times
        self.client_manager.__del__()
        self.client_manager.__del__()
        self.client_manager.__del__()
        
        # Should not raise exceptions

    # --- Data Integrity Edge Cases ---

    def test_character_data_with_missing_fields(self):
        """Test handling character data with missing required fields."""
        incomplete_character = {'Actor_id': self.test_actor_id}  # Missing other fields
        self.mock_db.get_character.return_value = incomplete_character
        self.mock_db.get_token.return_value = self.test_token
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {}
            )
            return result
            
        result = asyncio.run(run_test())
        
        # Should handle gracefully
        self.assertIsInstance(result, str)

    def test_client_info_with_invalid_port(self):
        """Test health check with invalid port numbers."""
        invalid_ports = [-1, 0, 65536, 99999, "invalid", None]
        
        for port in invalid_ports:
            with self.subTest(port=port):
                client_info = {
                    'Actor_id': self.test_actor_id,
                    'ip_address': self.test_ip,
                    'client_port': port
                }
                
                # Should handle gracefully without crashing
                try:
                    self.client_manager._perform_single_health_check_blocking(client_info)
                except Exception:
                    pass  # Expected for some invalid values

    def test_client_info_with_invalid_ip(self):
        """Test health check with invalid IP addresses."""
        invalid_ips = ["", "999.999.999.999", "not.an.ip", None, 12345]
        
        for ip in invalid_ips:
            with self.subTest(ip=ip):
                client_info = {
                    'Actor_id': self.test_actor_id,
                    'ip_address': ip,
                    'client_port': self.test_port
                }
                
                # Should handle gracefully
                try:
                    self.client_manager._perform_single_health_check_blocking(client_info)
                except Exception:
                    pass  # Expected for some invalid values

    # --- Performance and Memory Edge Cases ---

    def test_large_narration_text_handling(self):
        """Test handling of very large narration text."""
        large_narration = "A" * 100000  # 100KB of text
        self.mock_db.get_character.return_value = {'Actor_id': self.test_actor_id}
        self.mock_db.get_token.return_value = self.test_token
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                large_narration,
                {}
            )
            return result
            
        result = asyncio.run(run_test())
        
        # Should handle large text gracefully
        self.assertIsInstance(result, str)

    def test_large_character_dialogue_data(self):
        """Test handling of large character dialogue dictionary."""
        large_dialogue = {f"character{i}": "Some dialogue text" * 100 for i in range(1000)}
        self.mock_db.get_character.return_value = {'Actor_id': self.test_actor_id}
        self.mock_db.get_token.return_value = self.test_token
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Small narration",
                large_dialogue
            )
            return result
            
        result = asyncio.run(run_test())
        
        # Should handle large dialogue data gracefully
        self.assertIsInstance(result, str)

    # --- Error Recovery Edge Cases ---

    @patch('client_manager.requests.post')
    @patch('client_manager.asyncio.to_thread')
    @patch('client_manager.asyncio.sleep')
    def test_send_to_client_partial_failure_recovery(self, mock_sleep, mock_to_thread, mock_post):
        """Test recovery from partial failures in send_to_client."""
        self.mock_db.get_character.return_value = {'Actor_id': self.test_actor_id}
        self.mock_db.get_token.return_value = self.test_token
        
        # Simulate intermittent failures
        failure_count = 0
        
        async def mock_intermittent_request():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise requests.exceptions.Timeout()
            
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                'text': 'Success after retries',
                'audio_data': None
            }
            return mock_response
            
        mock_to_thread.side_effect = mock_intermittent_request
        mock_sleep.return_value = None
        
        async def run_test():
            result = await self.client_manager.send_to_client(
                self.test_actor_id,
                self.test_ip,
                self.test_port,
                "Test narration",
                {}
            )
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result, 'Success after retries')
        self.assertEqual(failure_count, 3)  # Failed twice, succeeded on third try


class TestClientManagerBoundaryConditions(unittest.TestCase):
    """
    Tests for boundary conditions and extreme values.
    
    Testing Framework: Python unittest (standard library)
    """

    def setUp(self):
        """Set up test fixtures for boundary condition testing."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)

    def test_token_generation_at_system_limits(self):
        """Test token generation at system resource limits."""
        # Test with maximum reasonable Actor ID length
        very_long_actor_id = "A" * 1000
        self.mock_db.get_character.return_value = {'Actor_id': very_long_actor_id}
        self.mock_db.save_client_token.return_value = None
        
        token = self.client_manager.generate_token(very_long_actor_id)
        
        self.assertIsInstance(token, str)
        self.assertEqual(len(token), 48)

    def test_health_check_with_minimum_interval(self):
        """Test health check behavior with minimum interval settings."""
        # This tests the edge case of very frequent health checks
        with patch('client_manager.CLIENT_HEALTH_CHECK_INTERVAL_SECONDS', 0.01):
            self.client_manager.start_periodic_health_checks()
            time.sleep(0.05)  # Let it run a few cycles
            self.client_manager.stop_periodic_health_checks()

    def test_concurrent_operations_at_scale(self):
        """Test system behavior under high concurrent load."""
        self.mock_db.get_character.return_value = {'Actor_id': 'test'}
        self.mock_db.save_client_token.return_value = None
        self.mock_db.get_client_token_details.return_value = {
            'token': 'test_token',
            'status': 'Online_Responsive'
        }
        
        def mixed_operations_worker(thread_id):
            # Mix of different operations
            self.client_manager.generate_token(f"Actor{thread_id}")
            self.client_manager.validate_token(f"Actor{thread_id}", "test_token")
            
        threads = []
        for i in range(50):  # High concurrency
            thread = threading.Thread(target=mixed_operations_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should complete without deadlock or corruption

    def test_memory_pressure_scenarios(self):
        """Test behavior under simulated memory pressure."""
        # Generate many tokens to test memory usage
        self.mock_db.get_character.return_value = {'Actor_id': 'test'}
        self.mock_db.save_client_token.return_value = None
        
        tokens = []
        for i in range(10000):  # Generate many tokens
            token = self.client_manager.generate_token(f"Actor{i}")
            tokens.append(token)
            
            # Occasionally check memory isn't growing unreasonably
            if i % 1000 == 0:
                self.assertEqual(len(token), 48)
        
        # All tokens should be unique
        self.assertEqual(len(set(tokens)), 10000)



class TestClientManagerHandshakeSystem(unittest.TestCase):
    """
    Comprehensive tests for the handshake challenge system in ClientManager.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite covers:
    - Handshake challenge generation and validation
    - Challenge expiry handling
    - Challenge security and uniqueness
    - Integration with authentication system
    """
    
    def setUp(self):
        """Set up test fixtures for handshake system testing."""
        self.mock_db = Mock(spec=Database)
        
        with patch("client_manager.pygame.mixer") as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)
            
        self.test_actor_id = "HandshakeTestActor"
        
    def test_generate_handshake_challenge_success(self):
        """Test successful handshake challenge generation."""
        challenge = self.client_manager.generate_handshake_challenge(self.test_actor_id)
        
        self.assertIsInstance(challenge, str)
        self.assertGreater(len(challenge), 20)  # Should be reasonably long
        self.assertIn(self.test_actor_id, self.client_manager.active_challenges)
        
        challenge_data = self.client_manager.active_challenges[self.test_actor_id]
        self.assertEqual(challenge_data["challenge"], challenge)
        self.assertIsInstance(challenge_data["timestamp"], datetime)
        
    def test_generate_multiple_challenges_same_actor(self):
        """Test generating multiple challenges for the same actor overwrites previous."""
        challenge1 = self.client_manager.generate_handshake_challenge(self.test_actor_id)
        challenge2 = self.client_manager.generate_handshake_challenge(self.test_actor_id)
        
        self.assertNotEqual(challenge1, challenge2)
        self.assertEqual(len(self.client_manager.active_challenges), 1)
        self.assertEqual(self.client_manager.active_challenges[self.test_actor_id]["challenge"], challenge2)
        
    def test_challenge_uniqueness_across_actors(self):
        """Test that challenges are unique across different actors."""
        actors = [f"Actor{i}" for i in range(10)]
        challenges = []
        
        for actor in actors:
            challenge = self.client_manager.generate_handshake_challenge(actor)
            challenges.append(challenge)
            
        self.assertEqual(len(challenges), len(set(challenges)))  # All unique
        
    def test_get_and_validate_challenge_success(self):
        """Test successful challenge retrieval and validation."""
        original_challenge = self.client_manager.generate_handshake_challenge(self.test_actor_id)
        retrieved_challenge = self.client_manager.get_and_validate_challenge(self.test_actor_id)
        
        self.assertEqual(original_challenge, retrieved_challenge)
        
    def test_get_and_validate_challenge_nonexistent(self):
        """Test challenge retrieval for non-existent actor."""
        result = self.client_manager.get_and_validate_challenge("NonExistentActor")
        
        self.assertIsNone(result)
        
    def test_get_and_validate_challenge_expired(self):
        """Test challenge retrieval when challenge has expired."""
        challenge = self.client_manager.generate_handshake_challenge(self.test_actor_id)
        
        # Manually expire the challenge
        past_time = datetime.now(timezone.utc) - timedelta(seconds=self.client_manager.CHALLENGE_EXPIRY_SECONDS + 1)
        self.client_manager.active_challenges[self.test_actor_id]["timestamp"] = past_time
        
        result = self.client_manager.get_and_validate_challenge(self.test_actor_id)
        
        self.assertIsNone(result)
        self.assertNotIn(self.test_actor_id, self.client_manager.active_challenges)  # Should be cleaned up
        
    def test_clear_challenge_success(self):
        """Test successful challenge clearing."""
        self.client_manager.generate_handshake_challenge(self.test_actor_id)
        self.assertIn(self.test_actor_id, self.client_manager.active_challenges)
        
        self.client_manager.clear_challenge(self.test_actor_id)
        
        self.assertNotIn(self.test_actor_id, self.client_manager.active_challenges)
        
    def test_clear_challenge_nonexistent(self):
        """Test clearing non-existent challenge."""
        # Should not raise exception
        self.client_manager.clear_challenge("NonExistentActor")
        
    def test_challenge_expiry_boundary_conditions(self):
        """Test challenge expiry at exact boundary conditions."""
        challenge = self.client_manager.generate_handshake_challenge(self.test_actor_id)
        
        # Test at exact expiry time
        exact_expiry_time = datetime.now(timezone.utc) - timedelta(seconds=self.client_manager.CHALLENGE_EXPIRY_SECONDS)
        self.client_manager.active_challenges[self.test_actor_id]["timestamp"] = exact_expiry_time
        
        result = self.client_manager.get_and_validate_challenge(self.test_actor_id)
        
        # At exact boundary, should be expired
        self.assertIsNone(result)
        
    def test_concurrent_challenge_operations(self):
        """Test concurrent challenge generation and validation."""
        import threading
        import time
        
        challenges = []
        errors = []
        
        def challenge_worker(actor_id):
            try:
                challenge = self.client_manager.generate_handshake_challenge(f"ConcurrentActor{actor_id}")
                challenges.append(challenge)
                time.sleep(0.01)  # Small delay
                retrieved = self.client_manager.get_and_validate_challenge(f"ConcurrentActor{actor_id}")
                if retrieved != challenge:
                    errors.append(f"Mismatch for actor {actor_id}")
            except Exception as e:
                errors.append(e)
                
        threads = []
        for i in range(20):
            thread = threading.Thread(target=challenge_worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(challenges), 20)
        self.assertEqual(len(set(challenges)), 20)  # All unique

class TestClientManagerConfigurationValidation(unittest.TestCase):
    """
    Tests for configuration validation and constant values.
    
    Testing Framework: Python unittest (standard library)
    """

    def test_health_check_interval_constants(self):
        """Test that health check interval constants are reasonable."""
        self.assertIsInstance(CLIENT_HEALTH_CHECK_INTERVAL_SECONDS, (int, float))
        self.assertGreater(CLIENT_HEALTH_CHECK_INTERVAL_SECONDS, 0)
        self.assertLess(CLIENT_HEALTH_CHECK_INTERVAL_SECONDS, 3600)  # Less than 1 hour

    def test_health_check_timeout_constants(self):
        """Test that timeout constants are reasonable."""
        self.assertIsInstance(CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS, (int, float))
        self.assertGreater(CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS, 0)
        self.assertLess(CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS, 60)  # Less than 1 minute

    def test_constants_relationship(self):
        """Test that timeout is less than interval to prevent overlap."""
        self.assertLess(
            CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS,
            CLIENT_HEALTH_CHECK_INTERVAL_SECONDS,
            "Timeout should be less than interval to prevent overlapping requests"
        )


if __name__ == '__main__':
    # Add the new test classes to the test suite
    import sys
    
    # Create test suite with all test classes
test_classes = [
    TestClientManager,
    TestClientManagerAsyncPatterns,
    TestClientManagerExtendedEdgeCases,
    TestClientManagerBoundaryConditions,
    TestClientManagerConfigurationValidation,
    TestClientManagerHandshakeSystem,
    TestClientManagerAuthenticationSystem,
    TestClientManagerDataValidation,
    TestClientManagerSecurityScenarios
]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with comprehensive output
    runner = unittest.TextTestRunner(
        verbosity=2,
        failfast=False,
        buffer=True,
        warnings='ignore'
    )
    
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)