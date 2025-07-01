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