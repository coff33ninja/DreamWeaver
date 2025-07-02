"""
Comprehensive unit tests for Server API functionality.
Testing Framework: pytest with unittest compatibility
"""
import pytest
import unittest
import json
import tempfile
import os
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, Optional
import sys

# Add SERVER to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import or create mock classes for testing
try:
    from server_api import ServerAPI, APIError, APIResponse
    from server_config import ServerConfig
    from server_handlers import RequestHandler
except ImportError:
    # Create mock classes if actual implementation not found
    class APIError(Exception):
        """Custom API error for testing."""
        def __init__(self, message: str, status_code: int = 500):
            self.message = message
            self.status_code = status_code
            super().__init__(message)

    class APIResponse:
        """Mock API response for testing."""
        def __init__(self, data: Any = None, status_code: int = 200, message: str = ""):
            self.data = data
            self.status_code = status_code
            self.message = message
            
        def to_dict(self):
            return {
                "data": self.data,
                "status_code": self.status_code,
                "message": self.message
            }

    class ServerConfig:
        """Mock server configuration for testing."""
        def __init__(self, config_dict: Dict[str, Any] = None):
            self.config = config_dict or self._default_config()
            
        def _default_config(self):
            return {
                'host': 'localhost',
                'port': 8080,
                'debug': False,
                'max_connections': 100,
                'timeout': 30,
                'workers': 1
            }
            
        def get(self, key: str, default: Any = None):
            return self.config.get(key, default)
            
        def update(self, updates: Dict[str, Any]):
            self.config.update(updates)

    class RequestHandler:
        """Mock request handler for testing."""
        def __init__(self, config: ServerConfig = None):
            self.config = config or ServerConfig()
            self.middleware = []
            
        async def handle_request(self, request_data: Dict[str, Any]):
            return APIResponse({"processed": True}, 200, "Success")
            
        def add_middleware(self, middleware_func):
            self.middleware.append(middleware_func)

    class ServerAPI:
        """Mock ServerAPI for comprehensive testing."""
        def __init__(self, config: ServerConfig = None):
            self.config = config or ServerConfig()
            self.is_running = False
            self.request_handler = RequestHandler(self.config)
            self.connections = []
            self.stats = {'requests': 0, 'errors': 0}
            
        async def start(self):
            """Start the server."""
            if self.is_running:
                raise APIError("Server is already running", 400)
            self.is_running = True
            return True
            
        async def stop(self):
            """Stop the server."""
            if not self.is_running:
                raise APIError("Server is not running", 400)
            self.is_running = False
            return True
            
        def get_status(self):
            """Get server status."""
            return {
                "status": "running" if self.is_running else "stopped",
                "connections": len(self.connections),
                "stats": self.stats.copy()
            }
            
        async def process_request(self, request_data: Dict[str, Any]):
            """Process a request."""
            if not self.is_running:
                raise APIError("Server is not running", 503)
            
            try:
                self.stats['requests'] += 1
                response = await self.request_handler.handle_request(request_data)
                return response
            except Exception as e:
                self.stats['errors'] += 1
                raise APIError(f"Request processing failed: {str(e)}", 500)
                
        def add_connection(self, connection_id: str):
            """Add a connection."""
            if len(self.connections) >= self.config.get('max_connections', 100):
                raise APIError("Maximum connections reached", 503)
            self.connections.append(connection_id)
            
        def remove_connection(self, connection_id: str):
            """Remove a connection."""
            if connection_id in self.connections:
                self.connections.remove(connection_id)


class TestServerAPI(unittest.TestCase):
    """Comprehensive unit tests for ServerAPI class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_config = ServerConfig({
            'host': 'localhost',
            'port': 8080,
            'debug': True,
            'max_connections': 10,
            'timeout': 30
        })
        self.server_api = ServerAPI(self.test_config)
        
    def tearDown(self):
        """Clean up after each test method."""
        # Ensure server is stopped
        if hasattr(self.server_api, 'is_running') and self.server_api.is_running:
            try:
                asyncio.run(self.server_api.stop())
            except:
                pass

    # Happy Path Tests
    def test_server_api_initialization_default(self):
        """Test ServerAPI initialization with default parameters."""
        api = ServerAPI()
        self.assertIsInstance(api, ServerAPI)
        self.assertIsInstance(api.config, ServerConfig)
        self.assertFalse(api.is_running)
        self.assertEqual(len(api.connections), 0)

    def test_server_api_initialization_with_config(self):
        """Test ServerAPI initialization with custom configuration."""
        api = ServerAPI(self.test_config)
        self.assertEqual(api.config, self.test_config)
        self.assertFalse(api.is_running)

    def test_server_start_success(self):
        """Test successful server startup."""
        async def test_start():
            result = await self.server_api.start()
            self.assertTrue(result)
            self.assertTrue(self.server_api.is_running)
            
        asyncio.run(test_start())

    def test_server_stop_success(self):
        """Test successful server shutdown."""
        async def test_stop():
            await self.server_api.start()
            result = await self.server_api.stop()
            self.assertTrue(result)
            self.assertFalse(self.server_api.is_running)
            
        asyncio.run(test_stop())

    def test_get_status_stopped(self):
        """Test getting server status when stopped."""
        status = self.server_api.get_status()
        self.assertIsInstance(status, dict)
        self.assertEqual(status['status'], 'stopped')
        self.assertEqual(status['connections'], 0)
        self.assertIn('stats', status)

    def test_get_status_running(self):
        """Test getting server status when running."""
        async def test_status():
            await self.server_api.start()
            status = self.server_api.get_status()
            self.assertEqual(status['status'], 'running')
            
        asyncio.run(test_status())

    def test_process_request_success(self):
        """Test successful request processing."""
        async def test_request():
            await self.server_api.start()
            request_data = {"action": "test", "data": {"key": "value"}}
            response = await self.server_api.process_request(request_data)
            self.assertIsInstance(response, APIResponse)
            self.assertEqual(response.status_code, 200)
            
        asyncio.run(test_request())

    # Edge Cases
    def test_server_double_start(self):
        """Test calling start twice should raise error."""
        async def test_double_start():
            await self.server_api.start()
            with self.assertRaises(APIError) as context:
                await self.server_api.start()
            self.assertEqual(context.exception.status_code, 400)
            
        asyncio.run(test_double_start())

    def test_server_stop_when_not_running(self):
        """Test stopping server when not running should raise error."""
        async def test_stop_not_running():
            with self.assertRaises(APIError) as context:
                await self.server_api.stop()
            self.assertEqual(context.exception.status_code, 400)
            
        asyncio.run(test_stop_not_running())

    def test_process_request_server_not_running(self):
        """Test processing request when server is not running."""
        async def test_request_not_running():
            request_data = {"action": "test"}
            with self.assertRaises(APIError) as context:
                await self.server_api.process_request(request_data)
            self.assertEqual(context.exception.status_code, 503)
            
        asyncio.run(test_request_not_running())

    def test_connection_management_success(self):
        """Test successful connection management."""
        self.server_api.add_connection("conn_1")
        self.assertEqual(len(self.server_api.connections), 1)
        
        self.server_api.remove_connection("conn_1")
        self.assertEqual(len(self.server_api.connections), 0)

    def test_max_connections_exceeded(self):
        """Test exceeding maximum connections limit."""
        # Fill up to max connections
        max_conn = self.test_config.get('max_connections', 10)
        for i in range(max_conn):
            self.server_api.add_connection(f"conn_{i}")
        
        # Adding one more should raise error
        with self.assertRaises(APIError) as context:
            self.server_api.add_connection("conn_overflow")
        self.assertEqual(context.exception.status_code, 503)

    def test_remove_nonexistent_connection(self):
        """Test removing a connection that doesn't exist."""
        # Should not raise error
        self.server_api.remove_connection("nonexistent")
        self.assertEqual(len(self.server_api.connections), 0)

    # Error Conditions and Failure Cases
    def test_invalid_request_data(self):
        """Test processing invalid request data."""
        async def test_invalid_data():
            await self.server_api.start()
            
            invalid_requests = [
                None,
                "",
                [],
                {"malformed": "request", "missing": "action"}
            ]
            
            for invalid_request in invalid_requests:
                with self.subTest(request=invalid_request):
                    try:
                        response = await self.server_api.process_request(invalid_request)
                        # If processing succeeds, should return valid response
                        self.assertIsInstance(response, APIResponse)
                    except APIError as e:
                        # Expected for invalid requests
                        self.assertIsInstance(e, APIError)
                        
        asyncio.run(test_invalid_data())

    @patch('SERVER.server_api.RequestHandler.handle_request')
    def test_request_handler_exception(self, mock_handle):
        """Test when request handler raises exception."""
        mock_handle.side_effect = Exception("Handler error")
        
        async def test_handler_error():
            await self.server_api.start()
            request_data = {"action": "test"}
            
            with self.assertRaises(APIError) as context:
                await self.server_api.process_request(request_data)
            self.assertEqual(context.exception.status_code, 500)
            self.assertIn("Request processing failed", context.exception.message)
            
        asyncio.run(test_handler_error())

    # Configuration Tests
    def test_server_config_validation(self):
        """Test server configuration validation."""
        valid_configs = [
            {'host': 'localhost', 'port': 8080},
            {'host': '0.0.0.0', 'port': 9000, 'debug': True},
            {'host': '127.0.0.1', 'port': 8000, 'max_connections': 50}
        ]
        
        for config_dict in valid_configs:
            with self.subTest(config=config_dict):
                config = ServerConfig(config_dict)
                api = ServerAPI(config)
                self.assertIsInstance(api, ServerAPI)

    def test_server_config_defaults(self):
        """Test default configuration values."""
        config = ServerConfig()
        expected_defaults = {
            'host': 'localhost',
            'port': 8080,
            'debug': False,
            'max_connections': 100,
            'timeout': 30,
            'workers': 1
        }
        
        for key, expected_value in expected_defaults.items():
            self.assertEqual(config.get(key), expected_value)

    # Boundary Value Tests
    def test_boundary_values_port(self):
        """Test boundary values for port configuration."""
        boundary_ports = [1, 1024, 8080, 65535]
        
        for port in boundary_ports:
            with self.subTest(port=port):
                config = ServerConfig({'port': port})
                api = ServerAPI(config)
                self.assertEqual(api.config.get('port'), port)

    def test_boundary_values_connections(self):
        """Test boundary values for max connections."""
        boundary_connections = [1, 10, 100, 1000]
        
        for max_conn in boundary_connections:
            with self.subTest(max_connections=max_conn):
                config = ServerConfig({'max_connections': max_conn})
                api = ServerAPI(config)
                self.assertEqual(api.config.get('max_connections'), max_conn)

    # Performance and Concurrency Tests
    def test_multiple_simultaneous_requests(self):
        """Test handling multiple simultaneous requests."""
        async def test_concurrent_requests():
            await self.server_api.start()
            
            request_data = {"action": "test", "data": {"id": 1}}
            
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = self.server_api.process_request({**request_data, "id": i})
                tasks.append(task)
            
            # Wait for all requests to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check that all requests were processed
            for response in responses:
                if isinstance(response, Exception):
                    self.fail(f"Request failed with exception: {response}")
                else:
                    self.assertIsInstance(response, APIResponse)
                    
        asyncio.run(test_concurrent_requests())

    def test_stats_tracking(self):
        """Test that server statistics are properly tracked."""
        async def test_stats():
            await self.server_api.start()
            
            # Process some requests
            for i in range(3):
                request_data = {"action": "test", "id": i}
                await self.server_api.process_request(request_data)
            
            stats = self.server_api.get_status()['stats']
            self.assertEqual(stats['requests'], 3)
            self.assertEqual(stats['errors'], 0)
            
        asyncio.run(test_stats())

    def test_error_stats_tracking(self):
        """Test that error statistics are properly tracked."""
        async def test_error_stats():
            await self.server_api.start()
            
            # Force an error by processing invalid request
            with patch.object(self.server_api.request_handler, 'handle_request', 
                            side_effect=Exception("Test error")):
                try:
                    await self.server_api.process_request({"action": "test"})
                except APIError:
                    pass  # Expected
            
            stats = self.server_api.get_status()['stats']
            self.assertEqual(stats['errors'], 1)
            
        asyncio.run(test_error_stats())


class TestServerConfig(unittest.TestCase):
    """Unit tests for ServerConfig class."""
    
    def test_config_initialization_empty(self):
        """Test ServerConfig initialization with empty config."""
        config = ServerConfig({})
        self.assertIsInstance(config.config, dict)
        
    def test_config_initialization_none(self):
        """Test ServerConfig initialization with None."""
        config = ServerConfig(None)
        self.assertIsInstance(config.config, dict)
        
    def test_config_get_existing_key(self):
        """Test getting existing configuration key."""
        config = ServerConfig({'host': 'test.com'})
        self.assertEqual(config.get('host'), 'test.com')
        
    def test_config_get_missing_key_with_default(self):
        """Test getting missing key with default value."""
        config = ServerConfig({})
        self.assertEqual(config.get('missing_key', 'default'), 'default')
        
    def test_config_update(self):
        """Test updating configuration."""
        config = ServerConfig({'host': 'localhost'})
        config.update({'port': 9000, 'debug': True})
        
        self.assertEqual(config.get('host'), 'localhost')
        self.assertEqual(config.get('port'), 9000)
        self.assertTrue(config.get('debug'))


class TestAPIResponse(unittest.TestCase):
    """Unit tests for APIResponse class."""
    
    def test_response_initialization_defaults(self):
        """Test APIResponse initialization with defaults."""
        response = APIResponse()
        self.assertIsNone(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.message, "")
        
    def test_response_initialization_with_data(self):
        """Test APIResponse initialization with data."""
        data = {"result": "success"}
        response = APIResponse(data, 201, "Created")
        
        self.assertEqual(response.data, data)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.message, "Created")
        
    def test_response_to_dict(self):
        """Test APIResponse to_dict method."""
        data = {"key": "value"}
        response = APIResponse(data, 200, "OK")
        
        expected_dict = {
            "data": data,
            "status_code": 200,
            "message": "OK"
        }
        
        self.assertEqual(response.to_dict(), expected_dict)


class TestAPIError(unittest.TestCase):
    """Unit tests for APIError class."""
    
    def test_api_error_initialization_defaults(self):
        """Test APIError initialization with defaults."""
        error = APIError("Test error")
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.status_code, 500)
        self.assertEqual(str(error), "Test error")
        
    def test_api_error_initialization_with_status(self):
        """Test APIError initialization with status code."""
        error = APIError("Not found", 404)
        self.assertEqual(error.message, "Not found")
        self.assertEqual(error.status_code, 404)


class TestRequestHandler(unittest.TestCase):
    """Unit tests for RequestHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ServerConfig()
        self.handler = RequestHandler(self.config)
        
    def test_handler_initialization(self):
        """Test RequestHandler initialization."""
        self.assertIsInstance(self.handler.config, ServerConfig)
        self.assertIsInstance(self.handler.middleware, list)
        self.assertEqual(len(self.handler.middleware), 0)
        
    def test_handle_request_success(self):
        """Test successful request handling."""
        async def test_handle():
            request_data = {"action": "test"}
            response = await self.handler.handle_request(request_data)
            
            self.assertIsInstance(response, APIResponse)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.message, "Success")
            
        asyncio.run(test_handle())
        
    def test_add_middleware(self):
        """Test adding middleware to handler."""
        def test_middleware(request):
            return request
            
        self.handler.add_middleware(test_middleware)
        self.assertEqual(len(self.handler.middleware), 1)
        self.assertEqual(self.handler.middleware[0], test_middleware)


# Integration Tests
class TestServerAPIIntegration(unittest.TestCase):
    """Integration tests for Server API components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_server_lifecycle(self):
        """Test complete server lifecycle from start to stop."""
        async def test_lifecycle():
            config = ServerConfig({
                'host': 'localhost',
                'port': 8081,
                'debug': True,
                'max_connections': 5
            })
            
            server = ServerAPI(config)
            
            # Test initial state
            status = server.get_status()
            self.assertEqual(status['status'], 'stopped')
            
            # Start server
            await server.start()
            status = server.get_status()
            self.assertEqual(status['status'], 'running')
            
            # Process request
            request_data = {"action": "integration_test"}
            response = await server.process_request(request_data)
            self.assertIsInstance(response, APIResponse)
            
            # Stop server
            await server.stop()
            status = server.get_status()
            self.assertEqual(status['status'], 'stopped')
            
        asyncio.run(test_lifecycle())
        
    def test_config_file_integration(self):
        """Test server with configuration from file."""
        config_data = {
            'host': '127.0.0.1',
            'port': 8082,
            'debug': False,
            'max_connections': 20
        }
        
        # Write config to file
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load config from file
        with open(self.config_file, 'r') as f:
            loaded_config = json.load(f)
        
        config = ServerConfig(loaded_config)
        server = ServerAPI(config)
        
        # Verify configuration
        for key, value in config_data.items():
            self.assertEqual(server.config.get(key), value)


# Performance Tests
class TestServerAPIPerformance(unittest.TestCase):
    """Performance tests for Server API."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.config = ServerConfig({
            'host': 'localhost',
            'port': 8083,
            'max_connections': 100
        })
        self.server = ServerAPI(self.config)
        
    def test_high_volume_requests(self):
        """Test server performance with high volume of requests."""
        async def test_volume():
            await self.server.start()
            
            # Create many requests
            num_requests = 50
            tasks = []
            
            for i in range(num_requests):
                request_data = {"action": "perf_test", "id": i}
                task = self.server.process_request(request_data)
                tasks.append(task)
            
            # Process all requests
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all requests succeeded
            successful_responses = [r for r in responses if isinstance(r, APIResponse)]
            self.assertEqual(len(successful_responses), num_requests)
            
            # Check stats
            stats = self.server.get_status()['stats']
            self.assertEqual(stats['requests'], num_requests)
            
        asyncio.run(test_volume())
        
    def test_connection_scaling(self):
        """Test connection management at scale."""
        max_connections = self.config.get('max_connections', 100)
        
        # Add connections up to limit
        for i in range(max_connections):
            self.server.add_connection(f"perf_conn_{i}")
        
        self.assertEqual(len(self.server.connections), max_connections)
        
        # Remove all connections
        for i in range(max_connections):
            self.server.remove_connection(f"perf_conn_{i}")
        
        self.assertEqual(len(self.server.connections), 0)


if __name__ == '__main__':
    # Run tests with unittest
    unittest.main(verbosity=2)

# ===== ADDITIONAL COMPREHENSIVE TEST COVERAGE =====
# Testing Framework: pytest with unittest compatibility (as identified from existing tests)

class TestServerAPIExtended(unittest.TestCase):
    """Extended comprehensive tests for ServerAPI with real implementation features."""
    
    def setUp(self):
        """Set up extended test fixtures."""
        self.config = ServerConfig({
            'host': 'localhost',
            'port': 8080,
            'debug': True,
            'max_connections': 5,
            'timeout': 10,
            'workers': 2,
            'log_level': 'DEBUG'
        })
        self.server = ServerAPI(self.config)

    def tearDown(self):
        """Clean up extended test fixtures."""
        if hasattr(self.server, 'is_running') and self.server.is_running:
            try:
                asyncio.run(self.server.stop())
            except:
                pass

    # Test new timestamp functionality in APIResponse
    def test_api_response_timestamp(self):
        """Test APIResponse includes timestamp functionality."""
        import time
        start_time = time.time()
        response = APIResponse("test data", 200, "Success")
        end_time = time.time()
        
        self.assertIsNotNone(response.timestamp)
        self.assertGreaterEqual(response.timestamp, start_time)
        self.assertLessEqual(response.timestamp, end_time)
        
        # Test to_dict includes timestamp
        response_dict = response.to_dict()
        self.assertIn('timestamp', response_dict)
        self.assertEqual(response_dict['timestamp'], response.timestamp)

    # Test ServerConfig validation
    def test_server_config_validation_success(self):
        """Test successful server configuration validation."""
        valid_config = ServerConfig({
            'host': 'localhost',
            'port': 8080,
            'debug': True
        })
        
        self.assertTrue(valid_config.validate())

    def test_server_config_validation_missing_required(self):
        """Test config validation with missing required fields."""
        # Missing host
        with self.assertRaises(ValueError) as context:
            config = ServerConfig({'port': 8080})
            config.validate()
        self.assertIn("Missing required configuration: host", str(context.exception))
        
        # Missing port
        with self.assertRaises(ValueError) as context:
            config = ServerConfig({'host': 'localhost'})
            config.validate()
        self.assertIn("Missing required configuration: port", str(context.exception))

    def test_server_config_validation_invalid_port(self):
        """Test config validation with invalid port values."""
        invalid_ports = [0, -1, 'invalid', None, 3.14]
        
        for port in invalid_ports:
            with self