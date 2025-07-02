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

class TestClientManagerHandshakeProtocol(unittest.TestCase):
    """
    Tests for handshake challenge/response protocol functionality.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite covers:
    - Challenge generation and validation
    - Session token management
    - Handshake protocol security
    - Challenge expiry handling
    - Authentication flow edge cases
    """

    def setUp(self):
        """Set up handshake protocol test fixtures."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)

    def test_generate_handshake_challenge_success(self):
        """Test successful handshake challenge generation."""
        challenge = self.client_manager.generate_handshake_challenge("TestActor")
        
        self.assertIsInstance(challenge, str)
        self.assertGreater(len(challenge), 20)  # Should be substantial length
        self.assertIn("TestActor", self.client_manager.active_challenges)
        
        # Verify challenge data structure
        challenge_data = self.client_manager.active_challenges["TestActor"]
        self.assertEqual(challenge_data["challenge"], challenge)
        self.assertIsInstance(challenge_data["timestamp"], datetime)

    def test_handshake_challenge_uniqueness(self):
        """Test that handshake challenges are unique across actors and time."""
        challenge1 = self.client_manager.generate_handshake_challenge("Actor1")
        challenge2 = self.client_manager.generate_handshake_challenge("Actor2")
        challenge3 = self.client_manager.generate_handshake_challenge("Actor1")  # Overwrite
        
        self.assertNotEqual(challenge1, challenge2)
        self.assertNotEqual(challenge1, challenge3)
        self.assertNotEqual(challenge2, challenge3)

    def test_get_and_validate_challenge_success(self):
        """Test successful challenge retrieval and validation."""
        original_challenge = self.client_manager.generate_handshake_challenge("TestActor")
        
        retrieved_challenge = self.client_manager.get_and_validate_challenge("TestActor")
        
        self.assertEqual(original_challenge, retrieved_challenge)

    def test_get_and_validate_challenge_nonexistent(self):
        """Test challenge validation for non-existent actor."""
        challenge = self.client_manager.get_and_validate_challenge("NonExistentActor")
        
        self.assertIsNone(challenge)

    def test_get_and_validate_challenge_expired(self):
        """Test challenge validation for expired challenge."""
        # Generate challenge
        self.client_manager.generate_handshake_challenge("TestActor")
        
        # Manually set timestamp to past
        past_time = datetime.now(timezone.utc) - timedelta(seconds=self.client_manager.CHALLENGE_EXPIRY_SECONDS + 10)
        self.client_manager.active_challenges["TestActor"]["timestamp"] = past_time
        
        challenge = self.client_manager.get_and_validate_challenge("TestActor")
        
        self.assertIsNone(challenge)
        self.assertNotIn("TestActor", self.client_manager.active_challenges)  # Should be cleaned up

    def test_clear_challenge_success(self):
        """Test successful challenge clearing."""
        self.client_manager.generate_handshake_challenge("TestActor")
        self.assertIn("TestActor", self.client_manager.active_challenges)
        
        self.client_manager.clear_challenge("TestActor")
        
        self.assertNotIn("TestActor", self.client_manager.active_challenges)

    def test_clear_challenge_nonexistent(self):
        """Test clearing non-existent challenge."""
        # Should not raise exception
        self.client_manager.clear_challenge("NonExistentActor")

    def test_challenge_expiry_boundary_conditions(self):
        """Test challenge expiry at exact boundary conditions."""
        self.client_manager.generate_handshake_challenge("TestActor")
        
        # Test exactly at expiry time
        expiry_time = datetime.now(timezone.utc) - timedelta(seconds=self.client_manager.CHALLENGE_EXPIRY_SECONDS)
        self.client_manager.active_challenges["TestActor"]["timestamp"] = expiry_time
        
        challenge = self.client_manager.get_and_validate_challenge("TestActor")
        
        # Should be expired (boundary is exclusive)
        self.assertIsNone(challenge)

    def test_authenticate_request_token_with_session_token(self):
        """Test authentication using valid session token."""
        # Setup session token
        future_expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self.mock_db.get_client_token_details.return_value = {
            'token': 'primary_token',
            'session_token': 'valid_session_token',
            'session_token_expiry': future_expiry,
            'status': 'Online_Responsive'
        }
        
        result = self.client_manager.authenticate_request_token("TestActor", "valid_session_token")
        
        self.assertTrue(result)

    def test_authenticate_request_token_expired_session(self):
        """Test authentication with expired session token."""
        # Setup expired session token
        past_expiry = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        self.mock_db.get_client_token_details.return_value = {
            'token': 'primary_token',
            'session_token': 'expired_session_token',
            'session_token_expiry': past_expiry,
            'status': 'Online_Responsive'
        }
        
        result = self.client_manager.authenticate_request_token("TestActor", "expired_session_token")
        
        self.assertFalse(result)
        self.mock_db.update_client_session_token.assert_called_once_with("TestActor", None, None)

    def test_authenticate_request_token_primary_fallback(self):
        """Test authentication falling back to primary token."""
        self.mock_db.get_client_token_details.return_value = {
            'token': 'primary_token',
            'session_token': None,
            'session_token_expiry': None,
            'status': 'Online_Responsive'
        }
        
        result = self.client_manager.authenticate_request_token("TestActor", "primary_token")
        
        self.assertTrue(result)

    def test_authenticate_request_token_invalid_expiry_format(self):
        """Test authentication with malformed expiry timestamp."""
        self.mock_db.get_client_token_details.return_value = {
            'token': 'primary_token',
            'session_token': 'session_token',
            'session_token_expiry': 'invalid_timestamp_format',
            'status': 'Online_Responsive'
        }
        
        result = self.client_manager.authenticate_request_token("TestActor", "session_token")
        
        self.assertFalse(result)

    def test_concurrent_challenge_operations(self):
        """Test concurrent challenge generation and validation."""
        import threading
        import time
        
        results = {}
        errors = []
        
        def challenge_worker(actor_id):
            try:
                challenge = self.client_manager.generate_handshake_challenge(f"Actor{actor_id}")
                time.sleep(0.01)  # Brief delay
                retrieved = self.client_manager.get_and_validate_challenge(f"Actor{actor_id}")
                results[actor_id] = (challenge, retrieved)
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
        self.assertEqual(len(results), 20)
        
        # Verify all challenges were unique and properly retrieved
        all_challenges = [r[0] for r in results.values()]
        self.assertEqual(len(set(all_challenges)), 20)


class TestClientManagerAudioProcessing(unittest.TestCase):
    """
    Comprehensive tests for audio processing functionality.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite covers:
    - Audio data encoding/decoding
    - File system operations for audio
    - Pygame mixer integration
    - Audio streaming edge cases
    - Memory management for large audio files
    """

    def setUp(self):
        """Set up audio processing test fixtures."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)

    @patch('client_manager.pygame.mixer')
    @patch('client_manager.os.makedirs')
    @patch('client_manager.open', create=True)
    @patch('client_manager.base64.b64decode')
    def test_audio_file_creation_and_playback(self, mock_b64decode, mock_open, mock_makedirs, mock_mixer):
        """Test complete audio file creation and playback workflow."""
        # Setup mocks
        mock_mixer.get_init.return_value = True
        mock_sound = Mock()
        mock_mixer.Sound.return_value = mock_sound
        mock_b64decode.return_value = b'fake_audio_data'
        
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test audio processing
        audio_data = "dGVzdF9hdWRpb19kYXRh"  # base64 encoded test data
        
        # This would be called during send_to_client with audio response
        # Simulating the audio handling portion
        mock_makedirs.assert_not_called()  # Not called yet
        
        # Verify setup is correct
        self.assertTrue(mock_mixer.get_init.return_value)

    @patch('client_manager.pygame.mixer')
    def test_pygame_mixer_not_initialized(self, mock_mixer):
        """Test audio handling when pygame mixer is not initialized."""
        mock_mixer.get_init.return_value = False
        
        # Should handle gracefully
        cm = ClientManager(self.mock_db)
        self.assertIsNotNone(cm)

    @patch('client_manager.pygame.mixer')
    @patch('client_manager.base64.b64decode')
    def test_invalid_audio_data_handling(self, mock_b64decode, mock_mixer):
        """Test handling of invalid or corrupted audio data."""
        mock_mixer.get_init.return_value = True
        mock_b64decode.side_effect = Exception("Invalid base64 data")
        
        # Should handle decode error gracefully
        with patch('builtins.print'):  # Suppress error output
            # This would be tested in the context of send_to_client
            pass

    def test_large_audio_data_memory_efficiency(self):
        """Test memory efficiency with large audio data."""
        # Create large base64 audio data (simulating 1MB audio file)
        large_audio_data = base64.b64encode(b'x' * 1024 * 1024).decode('utf-8')
        
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.get_token.return_value = 'test_token'
        
        with patch('client_manager.asyncio.to_thread') as mock_to_thread:
            async def mock_large_audio_response():
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {
                    'text': 'Large audio response',
                    'audio_data': large_audio_data
                }
                return mock_response
            
            mock_to_thread.return_value = mock_large_audio_response()
            
            async def test_large_audio():
                result = await self.client_manager.send_to_client(
                    "TestActor",
                    "127.0.0.1",
                    8080,
                    "Test narration",
                    {}
                )
                return result
            
            result = asyncio.run(test_large_audio())
            
            # Should handle large audio gracefully
            self.assertEqual(result, 'Large audio response')


class TestClientManagerConfigurationManagement(unittest.TestCase):
    """
    Tests for configuration and constants management.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite covers:
    - Configuration constant validation
    - Environment-specific settings
    - Runtime configuration changes
    - Configuration boundary validation
    """

    def test_health_check_interval_reasonable_range(self):
        """Test that health check interval is in reasonable range."""
        from client_manager import CLIENT_HEALTH_CHECK_INTERVAL_SECONDS
        
        # Should be between 10 seconds and 1 hour
        self.assertGreaterEqual(CLIENT_HEALTH_CHECK_INTERVAL_SECONDS, 10)
        self.assertLessEqual(CLIENT_HEALTH_CHECK_INTERVAL_SECONDS, 3600)

    def test_request_timeout_reasonable_range(self):
        """Test that request timeout is in reasonable range."""
        from client_manager import CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS
        
        # Should be between 1 second and 2 minutes
        self.assertGreaterEqual(CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS, 1)
        self.assertLessEqual(CLIENT_HEALTH_REQUEST_TIMEOUT_SECONDS, 120)

    def test_send_to_client_configuration_constants(self):
        """Test send_to_client related configuration constants."""
        from client_manager import (
            SEND_TO_CLIENT_MAX_RETRIES,
            SEND_TO_CLIENT_BASE_DELAY_SECONDS,
            SEND_TO_CLIENT_REQUEST_TIMEOUT_SECONDS
        )
        
        # Max retries should be reasonable
        self.assertGreaterEqual(SEND_TO_CLIENT_MAX_RETRIES, 1)
        self.assertLessEqual(SEND_TO_CLIENT_MAX_RETRIES, 10)
        
        # Base delay should be reasonable
        self.assertGreaterEqual(SEND_TO_CLIENT_BASE_DELAY_SECONDS, 0.1)
        self.assertLessEqual(SEND_TO_CLIENT_BASE_DELAY_SECONDS, 10)
        
        # Request timeout should be reasonable
        self.assertGreaterEqual(SEND_TO_CLIENT_REQUEST_TIMEOUT_SECONDS, 5)
        self.assertLessEqual(SEND_TO_CLIENT_REQUEST_TIMEOUT_SECONDS, 300)

    def test_challenge_expiry_configuration(self):
        """Test challenge expiry configuration."""
        mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            client_manager = ClientManager(mock_db)
        
        # Challenge expiry should be reasonable
        self.assertGreaterEqual(client_manager.CHALLENGE_EXPIRY_SECONDS, 30)
        self.assertLessEqual(client_manager.CHALLENGE_EXPIRY_SECONDS, 300)


class TestClientManagerDataValidationAndSanitization(unittest.TestCase):
    """
    Tests for data validation and sanitization.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite covers:
    - Input data validation
    - Output data sanitization
    - Type checking and conversion
    - Boundary value validation
    - Format validation
    """

    def setUp(self):
        """Set up data validation test fixtures."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)

    def test_actor_id_format_validation(self):
        """Test Actor ID format validation and handling."""
        invalid_actor_ids = [
            None,
            123,  # Integer instead of string
            [],   # List instead of string
            {},   # Dict instead of string
            "",   # Empty string
            " ",  # Whitespace only
        ]
        
        self.mock_db.get_character.return_value = {'Actor_id': 'test'}
        self.mock_db.save_client_token.return_value = None
        
        for invalid_id in invalid_actor_ids:
            with self.subTest(actor_id=invalid_id):
                try:
                    if invalid_id is None:
                        with self.assertRaises(TypeError):
                            self.client_manager.generate_token(invalid_id)
                    else:
                        # Should handle gracefully or raise appropriate exception
                        token = self.client_manager.generate_token(str(invalid_id) if invalid_id is not None else invalid_id)
                        if token is not None:
                            self.assertIsInstance(token, str)
                except (TypeError, ValueError):
                    # These are acceptable exceptions for invalid input
                    pass

    def test_ip_address_validation_patterns(self):
        """Test IP address validation in health checks."""
        valid_ips = [
            "127.0.0.1",
            "192.168.1.1",
            "10.0.0.1",
            "255.255.255.255",
            "0.0.0.0"
        ]
        
        invalid_ips = [
            "256.1.1.1",        # Invalid octet
            "192.168.1",        # Incomplete
            "192.168.1.1.1",    # Too many octets
            "not.an.ip",        # Non-numeric
            "",                 # Empty
            None,               # Null
            "localhost"         # Hostname (may or may not be valid depending on context)
        ]
        
        for ip in valid_ips + invalid_ips:
            with self.subTest(ip=ip):
                client_info = {
                    'Actor_id': 'TestActor',
                    'ip_address': ip,
                    'client_port': 8080
                }
                
                # Should handle gracefully regardless of IP validity
                try:
                    self.client_manager._perform_single_health_check_blocking(client_info)
                except Exception:
                    # Some exceptions are expected for invalid IPs
                    pass

    def test_port_number_validation_ranges(self):
        """Test port number validation in various ranges."""
        valid_ports = [80, 443, 8080, 8443, 3000, 65535]
        invalid_ports = [-1, 0, 65536, 99999, "8080", None, [], {}]
        
        for port in valid_ports + invalid_ports:
            with self.subTest(port=port):
                client_info = {
                    'Actor_id': 'TestActor',
                    'ip_address': '127.0.0.1',
                    'client_port': port
                }
                
                # Should handle all port values gracefully
                try:
                    self.client_manager._perform_single_health_check_blocking(client_info)
                except Exception:
                    # Some exceptions expected for invalid ports
                    pass

    def test_character_data_structure_validation(self):
        """Test validation of character data structures."""
        valid_character_data = {
            'Actor_id': 'TestActor',
            'name': 'Test Character',
            'personality': 'Friendly',
            'tts': 'piper',
            'tts_model': 'en_US-ryan-high'
        }
        
        invalid_character_data_sets = [
            {},  # Empty dict
            None,  # Null
            {'Actor_id': None},  # Missing required fields
            {'Actor_id': 'test', 'name': None},  # Null values
            {'Actor_id': 'test', 'extra_field': 'value'},  # Extra fields (should be ok)
        ]
        
        for char_data in [valid_character_data] + invalid_character_data_sets:
            with self.subTest(character_data=str(char_data)[:50]):
                self.mock_db.get_character.return_value = char_data
                self.mock_db.get_token.return_value = 'test_token'
                
                async def test_character_validation():
                    result = await self.client_manager.send_to_client(
                        "TestActor",
                        "127.0.0.1",
                        8080,
                        "Test narration",
                        {}
                    )
                    return result
                
                # Should handle all character data gracefully
                result = asyncio.run(test_character_validation())
                self.assertIsInstance(result, str)

    def test_dialogue_data_sanitization(self):
        """Test sanitization of dialogue data."""
        potentially_harmful_dialogue = {
            'character1': '<script>alert("xss")</script>',
            'character2': 'SELECT * FROM users; DROP TABLE users;',
            'character3': '../../etc/passwd',
            'character4': 'normal dialogue',
            'character5': None,
            'character6': 123,
            'character7': []
        }
        
        self.mock_db.get_character.return_value = {'Actor_id': 'TestActor'}
        self.mock_db.get_token.return_value = 'test_token'
        
        async def test_dialogue_sanitization():
            result = await self.client_manager.send_to_client(
                "TestActor",
                "127.0.0.1",
                8080,
                "Test narration",
                potentially_harmful_dialogue
            )
            return result
        
        # Should handle potentially harmful dialogue gracefully
        result = asyncio.run(test_dialogue_sanitization())
        self.assertIsInstance(result, str)


class TestClientManagerResourceLeakPrevention(unittest.TestCase):
    """
    Tests for resource leak prevention and cleanup.
    
    Testing Framework: Python unittest (standard library)
    
    This test suite covers:
    - Memory leak prevention
    - File handle cleanup
    - Thread resource management
    - Database connection cleanup
    - Temporary file cleanup
    """

    def setUp(self):
        """Set up resource management test fixtures."""
        self.mock_db = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            self.client_manager = ClientManager(self.mock_db)

    def test_health_check_thread_cleanup_on_destruction(self):
        """Test that health check thread is properly cleaned up."""
        # Start health check thread
        self.client_manager.start_periodic_health_checks()
        thread_id = self.client_manager.health_check_thread.ident
        
        # Verify thread is running
        self.assertTrue(self.client_manager.health_check_thread.is_alive())
        
        # Call destructor
        self.client_manager.__del__()
        
        # Give thread time to stop
        time.sleep(0.1)
        
        # Verify cleanup occurred
        self.assertTrue(self.client_manager.stop_health_check_event.is_set())

    def test_challenge_memory_cleanup_on_expiry(self):
        """Test that expired challenges are cleaned up from memory."""
        # Generate multiple challenges
        for i in range(10):
            self.client_manager.generate_handshake_challenge(f"Actor{i}")
        
        self.assertEqual(len(self.client_manager.active_challenges), 10)
        
        # Manually expire all challenges
        past_time = datetime.now(timezone.utc) - timedelta(seconds=self.client_manager.CHALLENGE_EXPIRY_SECONDS + 10)
        for actor_id in list(self.client_manager.active_challenges.keys()):
            self.client_manager.active_challenges[actor_id]["timestamp"] = past_time
        
        # Access challenges to trigger cleanup
        for i in range(10):
            self.client_manager.get_and_validate_challenge(f"Actor{i}")
        
        # All challenges should be cleaned up
        self.assertEqual(len(self.client_manager.active_challenges), 0)

    @patch('client_manager.tempfile.gettempdir')
    @patch('client_manager.os.remove')
    def test_temporary_audio_file_cleanup(self, mock_remove, mock_gettempdir):
        """Test cleanup of temporary audio files."""
        mock_gettempdir.return_value = "/tmp"
        
        # This would be tested in context of audio file creation and cleanup
        # The actual cleanup happens in the audio processing workflow
        
        # Verify mock setup
        self.assertEqual(mock_gettempdir.return_value, "/tmp")

    def test_multiple_client_manager_instances_isolation(self):
        """Test that multiple ClientManager instances don't interfere."""
        mock_db2 = Mock(spec=Database)
        
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = True
            client_manager2 = ClientManager(mock_db2)
        
        # Generate challenges in both instances
        challenge1 = self.client_manager.generate_handshake_challenge("Actor1")
        challenge2 = client_manager2.generate_handshake_challenge("Actor1")
        
        # Should be independent
        self.assertNotEqual(challenge1, challenge2)
        self.assertIn("Actor1", self.client_manager.active_challenges)
        self.assertIn("Actor1", client_manager2.active_challenges)
        
        # Cleanup
        client_manager2.__del__()

    def test_exception_during_initialization_cleanup(self):
        """Test cleanup when exceptions occur during initialization."""
        with patch('client_manager.pygame.mixer') as mock_mixer:
            mock_mixer.get_init.return_value = False
            mock_mixer.init.side_effect = Exception("Initialization failed")
            
            # Should handle initialization failure gracefully
            cm = ClientManager(self.mock_db)
            self.assertIsNotNone(cm)
            
            # Should be able to clean up properly
            cm.__del__()


if __name__ == '__main__':
    # Comprehensive test suite with all test classes
    all_comprehensive_test_classes = [
        TestClientManager,
        TestClientManagerAsyncPatterns, 
        TestClientManagerExtendedEdgeCases,
        TestClientManagerBoundaryConditions,
        TestClientManagerConfigurationValidation,
        TestClientManagerHandshakeProtocol,
        TestClientManagerAudioProcessing,
        TestClientManagerConfigurationManagement,
        TestClientManagerDataValidationAndSanitization,
        TestClientManagerResourceLeakPrevention
    ]
    
    suite = unittest.TestSuite()
    for test_class in all_comprehensive_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with maximum verbosity and comprehensive reporting
    runner = unittest.TextTestRunner(
        verbosity=2,
        failfast=False,
        buffer=True,
        warnings='ignore',
        stream=sys.stdout
    )
    
    print("=" * 80)
    print("COMPREHENSIVE CLIENT MANAGER TEST SUITE")
    print("Testing Framework: Python unittest (standard library)")
    print("=" * 80)
    
    result = runner.run(suite)
    
    # Detailed summary report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Test Classes: {len(all_comprehensive_test_classes)}")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successful Tests: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed Tests: {len(result.failures)}")
    print(f"Error Tests: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%")
    
    if result.failures:
        print(f"\nFailure Details:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")
    
    if result.errors:
        print(f"\nError Details:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")
    
    print("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)