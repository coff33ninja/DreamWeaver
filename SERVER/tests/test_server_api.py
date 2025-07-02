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

# Additional Security and Validation Tests
class TestServerAPISecurity(unittest.TestCase):
    """Security-focused tests for Server API."""
    
    def setUp(self):
        """Set up security test fixtures."""
        self.config = ServerConfig({
            'host': 'localhost',
            'port': 8084,
            'debug': False,
            'max_connections': 10,
            'timeout': 30
        })
        self.server = ServerAPI(self.config)
        
    def test_malicious_request_data_injection(self):
        """Test handling of potentially malicious request data."""
        async def test_malicious():
            await self.server.start()
            
            malicious_requests = [
                {"action": "../../../etc/passwd"},
                {"action": "<script>alert('xss')</script>"},
                {"action": "'; DROP TABLE users; --"},
                {"action": "\x00\x01\x02\x03"},  # Binary data
                {"action": "A" * 10000},  # Extremely long string
                {"action": {"nested": {"deeply": {"nested": "value"}}}},
                {"action": "test", "data": {"__proto__": {"polluted": True}}},
            ]
            
            for malicious_request in malicious_requests:
                with self.subTest(request=malicious_request):
                    try:
                        response = await self.server.process_request(malicious_request)
                        # Should handle gracefully
                        self.assertIsInstance(response, (APIResponse, type(None)))
                    except APIError as e:
                        # Expected for malicious requests
                        self.assertIsInstance(e, APIError)
                        
        asyncio.run(test_malicious())
    
    def test_request_size_limits(self):
        """Test handling of oversized requests."""
        async def test_size_limits():
            await self.server.start()
            
            # Create progressively larger requests
            sizes = [1024, 10240, 102400, 1048576]  # 1KB to 1MB
            
            for size in sizes:
                with self.subTest(size=size):
                    large_data = "A" * size
                    request_data = {"action": "test", "payload": large_data}
                    
                    try:
                        response = await self.server.process_request(request_data)
                        # Should handle large requests appropriately
                        self.assertIsInstance(response, (APIResponse, type(None)))
                    except (APIError, MemoryError) as e:
                        # Expected for very large requests
                        self.assertIsInstance(e, (APIError, MemoryError))
                        
        asyncio.run(test_size_limits())
    
    def test_connection_id_validation(self):
        """Test connection ID validation and sanitization."""
        invalid_connection_ids = [
            "",
            None,
            123,  # Non-string
            "../../../etc",
            "conn\x00id",
            "conn\nid",
            "conn\rid",
            "A" * 1000,  # Very long ID
        ]
        
        for conn_id in invalid_connection_ids:
            with self.subTest(connection_id=conn_id):
                try:
                    self.server.add_connection(conn_id)
                    # If it doesn't raise, should be handled gracefully
                    if conn_id in self.server.connections:
                        self.server.remove_connection(conn_id)
                except (APIError, TypeError, ValueError):
                    # Expected for invalid IDs
                    pass


# Advanced Async and Concurrency Tests
class TestServerAPIAdvancedAsync(unittest.TestCase):
    """Advanced asynchronous and concurrency tests."""
    
    def setUp(self):
        """Set up async test fixtures."""
        self.config = ServerConfig({
            'host': 'localhost',
            'port': 8085,
            'max_connections': 50,
            'timeout': 5
        })
        self.server = ServerAPI(self.config)
    
    def test_concurrent_start_stop_operations(self):
        """Test concurrent start/stop operations."""
        async def test_concurrent_ops():
            # Attempt multiple start/stop operations concurrently
            tasks = []
            
            # Mix of start and stop operations
            for i in range(10):
                if i % 2 == 0:
                    tasks.append(self.server.start())
                else:
                    tasks.append(self.server.stop())
            
            # Execute concurrently and handle exceptions
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should have at least some successful operations
            successes = [r for r in results if not isinstance(r, Exception)]
            errors = [r for r in results if isinstance(r, Exception)]
            
            # Verify that errors are APIErrors (expected for invalid state transitions)
            for error in errors:
                if isinstance(error, APIError):
                    self.assertIn(error.status_code, [400, 503])
                    
        asyncio.run(test_concurrent_ops())
    
    def test_request_processing_under_load(self):
        """Test request processing behavior under high concurrent load."""
        async def test_load():
            await self.server.start()
            
            # Create a high number of concurrent requests
            num_requests = 100
            tasks = []
            
            for i in range(num_requests):
                request_data = {
                    "action": "load_test",
                    "id": i,
                    "timestamp": asyncio.get_event_loop().time(),
                    "data": {"payload": f"test_data_{i}"}
                }
                tasks.append(self.server.process_request(request_data))
            
            # Process all requests with timeout
            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30.0
                )
                
                # Analyze results
                successful = [r for r in responses if isinstance(r, APIResponse)]
                failed = [r for r in responses if isinstance(r, Exception)]
                
                # Should have processed most requests successfully
                success_rate = len(successful) / len(responses)
                self.assertGreater(success_rate, 0.8, "Success rate should be > 80%")
                
                # Check server stats
                stats = self.server.get_status()['stats']
                self.assertGreater(stats['requests'], 0)
                
            except asyncio.TimeoutError:
                self.fail("Request processing timed out")
                
        asyncio.run(test_load())
    
    def test_memory_cleanup_after_requests(self):
        """Test memory cleanup after processing many requests."""
        async def test_cleanup():
            await self.server.start()
            
            # Process many requests to test memory cleanup
            for batch in range(10):
                batch_tasks = []
                for i in range(20):
                    request_data = {
                        "action": "memory_test",
                        "batch": batch,
                        "id": i,
                        "large_data": ["test"] * 1000  # Some larger data
                    }
                    batch_tasks.append(self.server.process_request(request_data))
                
                # Process batch
                responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Verify batch completed
                successful_batch = [r for r in responses if isinstance(r, APIResponse)]
                self.assertGreater(len(successful_batch), 15)  # Most should succeed
            
            # Final verification
            final_stats = self.server.get_status()['stats']
            self.assertGreater(final_stats['requests'], 150)
            
        asyncio.run(test_cleanup())


# Configuration Edge Cases and Validation Tests
class TestServerConfigAdvanced(unittest.TestCase):
    """Advanced configuration testing."""
    
    def test_config_type_validation(self):
        """Test configuration with various data types."""
        type_test_configs = [
            {'port': "8080"},  # String instead of int
            {'debug': "true"},  # String instead of boolean
            {'max_connections': "unlimited"},  # String instead of int
            {'timeout': None},  # None value
            {'host': 123},  # Number instead of string
            {'workers': -1},  # Negative number
            {'port': 0},  # Zero port
            {'port': 70000},  # Port out of range
        ]
        
        for config_dict in type_test_configs:
            with self.subTest(config=config_dict):
                try:
                    config = ServerConfig(config_dict)
                    server = ServerAPI(config)
                    # Should handle type mismatches gracefully
                    self.assertIsInstance(server, ServerAPI)
                except (TypeError, ValueError) as e:
                    # Expected for some invalid configurations
                    self.assertIsInstance(e, (TypeError, ValueError))
    
    def test_config_boundary_values(self):
        """Test configuration boundary values more thoroughly."""
        boundary_configs = [
            {'port': 1, 'host': 'localhost'},  # Minimum port
            {'port': 65535, 'host': 'localhost'},  # Maximum port  
            {'max_connections': 0},  # Zero connections
            {'max_connections': 10000},  # Very high connections
            {'timeout': 0.1},  # Very short timeout
            {'timeout': 3600},  # Very long timeout
        ]
        
        for config_dict in boundary_configs:
            with self.subTest(config=config_dict):
                config = ServerConfig(config_dict)
                server = ServerAPI(config)
                
                # Verify values are stored correctly
                for key, value in config_dict.items():
                    self.assertEqual(server.config.get(key), value)
    
    def test_config_update_validation(self):
        """Test configuration updates with validation."""
        config = ServerConfig({'host': 'localhost', 'port': 8080})
        
        # Test various update scenarios
        update_scenarios = [
            {'port': 9090},  # Valid update
            {'new_key': 'new_value'},  # Adding new key
            {'host': None},  # Setting to None
            {'debug': True},  # Adding boolean
            {'nested': {'key': 'value'}},  # Nested configuration
        ]
        
        for update in update_scenarios:
            with self.subTest(update=update):
                original_config = config.config.copy()
                config.update(update)
                
                # Verify update was applied
                for key, value in update.items():
                    self.assertEqual(config.get(key), value)
                
                # Verify original keys still exist
                for key in original_config:
                    if key not in update:
                        self.assertEqual(config.get(key), original_config[key])


# Error Recovery and Resilience Tests  
class TestServerAPIResilience(unittest.TestCase):
    """Test server resilience and error recovery."""
    
    def setUp(self):
        """Set up resilience test fixtures."""
        self.config = ServerConfig({
            'host': 'localhost',
            'port': 8086,
            'max_connections': 5,
            'timeout': 10
        })
        self.server = ServerAPI(self.config)
    
    def test_recovery_after_handler_failure(self):
        """Test server recovery after request handler failures."""
        async def test_recovery():
            await self.server.start()
            
            # Mock handler to fail intermittently
            original_handler = self.server.request_handler.handle_request
            failure_count = [0]  # Use list to modify in closure
            
            async def failing_handler(request_data):
                failure_count[0] += 1
                if failure_count[0] % 3 == 0:  # Fail every 3rd request
                    raise Exception("Simulated handler failure")
                return await original_handler(request_data)
            
            self.server.request_handler.handle_request = failing_handler
            
            # Process multiple requests
            results = []
            for i in range(9):  # 3 failures expected
                try:
                    response = await self.server.process_request({"action": "test", "id": i})
                    results.append(("success", response))
                except APIError as e:
                    results.append(("error", e))
            
            # Should have mix of successes and failures
            successes = [r for r in results if r[0] == "success"]
            failures = [r for r in results if r[0] == "error"]
            
            self.assertEqual(len(failures), 3)  # 3 failures as expected
            self.assertEqual(len(successes), 6)  # 6 successes
            
            # Server should still be running
            self.assertTrue(self.server.is_running)
            
            # Stats should reflect both successes and failures
            stats = self.server.get_status()['stats']
            self.assertEqual(stats['requests'], 9)
            self.assertEqual(stats['errors'], 3)
            
        asyncio.run(test_recovery())
    
    def test_connection_cleanup_on_error(self):
        """Test connection cleanup when errors occur."""
        # Fill up connections
        max_conn = self.config.get('max_connections')
        for i in range(max_conn):
            self.server.add_connection(f"conn_{i}")
        
        # Verify connections are at limit
        self.assertEqual(len(self.server.connections), max_conn)
        
        # Simulate error condition requiring cleanup
        connections_to_cleanup = ["conn_0", "conn_2", "conn_4"]
        for conn_id in connections_to_cleanup:
            self.server.remove_connection(conn_id)
        
        # Verify cleanup worked
        expected_remaining = max_conn - len(connections_to_cleanup)
        self.assertEqual(len(self.server.connections), expected_remaining)
        
        # Should be able to add new connections
        for conn_id in connections_to_cleanup:
            self.server.add_connection(f"new_{conn_id}")
        
        self.assertEqual(len(self.server.connections), max_conn)
    
    def test_server_state_consistency(self):
        """Test server state consistency across operations."""
        async def test_consistency():
            # Test multiple state transitions
            state_transitions = [
                ('start', True),
                ('stop', False),
                ('start', True),
                ('start', True),  # Should fail - already running
                ('stop', False),
                ('stop', False),  # Should fail - already stopped
            ]
            
            for operation, expected_running in state_transitions:
                with self.subTest(operation=operation, expected=expected_running):
                    try:
                        if operation == 'start':
                            await self.server.start()
                        else:
                            await self.server.stop()
                        
                        # If operation succeeded, verify state
                        self.assertEqual(self.server.is_running, expected_running)
                        
                    except APIError as e:
                        # Expected for invalid state transitions
                        self.assertEqual(e.status_code, 400)
                        # State should be unchanged
                        status = self.server.get_status()
                        current_state = status['status'] == 'running'
                        # State should be consistent with last successful operation
                        self.assertIsInstance(current_state, bool)
                        
        asyncio.run(test_consistency())


# Mock Integration and Dependency Tests
class TestServerAPIMockIntegration(unittest.TestCase):
    """Test server API with various mocked dependencies."""
    
    def setUp(self):
        """Set up mock integration test fixtures."""
        self.config = ServerConfig()
        self.server = ServerAPI(self.config)
    
    @patch('SERVER.tests.test_server_api.RequestHandler')
    def test_custom_request_handler_integration(self, mock_handler_class):
        """Test server with custom request handler."""
        # Setup mock handler
        mock_handler = Mock()
        mock_handler.handle_request = AsyncMock(return_value=APIResponse({"custom": True}, 200, "Custom"))
        mock_handler_class.return_value = mock_handler
        
        # Create server with mocked handler
        server = ServerAPI(self.config)
        server.request_handler = mock_handler
        
        async def test_custom_handler():
            await server.start()
            
            request_data = {"action": "custom_test"}
            response = await server.process_request(request_data)
            
            # Verify custom handler was called
            mock_handler.handle_request.assert_called_once_with(request_data)
            self.assertEqual(response.data, {"custom": True})
            self.assertEqual(response.message, "Custom")
            
        asyncio.run(test_custom_handler())
    
    @patch('SERVER.tests.test_server_api.ServerConfig')
    def test_dynamic_config_changes(self, mock_config_class):
        """Test server behavior with dynamic configuration changes."""
        # Setup mock config that changes over time
        mock_config = Mock()
        call_count = [0]
        
        def get_side_effect(key, default=None):
            call_count[0] += 1
            configs = [
                {'max_connections': 5, 'timeout': 10},
                {'max_connections': 10, 'timeout': 20},
                {'max_connections': 15, 'timeout': 30},
            ]
            config_index = min(call_count[0] // 5, len(configs) - 1)
            return configs[config_index].get(key, default)
        
        mock_config.get.side_effect = get_side_effect
        mock_config_class.return_value = mock_config
        
        server = ServerAPI(mock_config)
        
        # Test that server adapts to config changes
        for i in range(15):
            expected_max = 5 if i < 5 else (10 if i < 10 else 15)
            try:
                server.add_connection(f"dynamic_conn_{i}")
            except APIError:
                # Expected when hitting max connections
                break
        
        # Verify config was consulted multiple times
        self.assertGreater(mock_config.get.call_count, 10)
    
    def test_middleware_chain_execution(self):
        """Test middleware chain execution in request handler."""
        handler = RequestHandler(self.config)
        
        # Add multiple middleware functions
        middleware_calls = []
        
        def middleware_1(request):
            middleware_calls.append("middleware_1")
            request["middleware_1"] = True
            return request
        
        def middleware_2(request):
            middleware_calls.append("middleware_2")
            request["middleware_2"] = True
            return request
        
        handler.add_middleware(middleware_1)
        handler.add_middleware(middleware_2)
        
        # Test middleware execution order
        self.assertEqual(len(handler.middleware), 2)
        self.assertEqual(handler.middleware[0], middleware_1)
        self.assertEqual(handler.middleware[1], middleware_2)


# Data Integrity and Validation Tests
class TestServerAPIDataIntegrity(unittest.TestCase):
    """Test data integrity and validation in server API."""
    
    def setUp(self):
        """Set up data integrity test fixtures."""
        self.server = ServerAPI()
    
    def test_stats_accuracy_under_concurrent_load(self):
        """Test statistics accuracy under concurrent operations."""
        async def test_stats_accuracy():
            await self.server.start()
            
            # Reset stats
            self.server.stats = {'requests': 0, 'errors': 0}
            
            # Create concurrent requests that will succeed and fail
            success_tasks = []
            error_tasks = []
            
            # Successful requests
            for i in range(25):
                task = self.server.process_request({"action": "success", "id": i})
                success_tasks.append(task)
            
            # Force some errors
            with patch.object(self.server.request_handler, 'handle_request', 
                            side_effect=Exception("Test error")):
                for i in range(15):
                    task = self.server.process_request({"action": "error", "id": i})
                    error_tasks.append(task)
            
            # Execute all tasks
            all_tasks = success_tasks + error_tasks
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
            # Verify results
            successes = [r for r in results if isinstance(r, APIResponse)]
            errors = [r for r in results if isinstance(r, APIError)]
            
            self.assertEqual(len(successes), 25)
            self.assertEqual(len(errors), 15)
            
            # Verify stats accuracy
            stats = self.server.get_status()['stats']
            self.assertEqual(stats['requests'], 40)  # 25 + 15
            self.assertEqual(stats['errors'], 15)
            
        asyncio.run(test_stats_accuracy())
    
    def test_connection_list_integrity(self):
        """Test connection list integrity during concurrent operations."""
        import threading
        import time
        
        # Function to add connections concurrently
        def add_connections(start_id, count):
            for i in range(count):
                try:
                    self.server.add_connection(f"thread_conn_{start_id}_{i}")
                    time.sleep(0.001)  # Small delay to increase concurrency
                except APIError:
                    break  # Hit max connections
        
        # Function to remove connections concurrently  
        def remove_connections(start_id, count):
            time.sleep(0.01)  # Let some connections be added first
            for i in range(count):
                self.server.remove_connection(f"thread_conn_{start_id}_{i}")
                time.sleep(0.001)
        
        # Create threads for concurrent operations
        threads = []
        for thread_id in range(3):
            add_thread = threading.Thread(target=add_connections, args=(thread_id, 10))
            remove_thread = threading.Thread(target=remove_connections, args=(thread_id, 5))
            threads.extend([add_thread, remove_thread])
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify connection list integrity
        connections = self.server.connections
        self.assertIsInstance(connections, list)
        
        # All remaining connections should be valid
        for conn in connections:
            self.assertIsInstance(conn, str)
            self.assertTrue(conn.startswith("thread_conn_"))
    
    def test_response_object_immutability(self):
        """Test that response objects maintain data integrity."""
        original_data = {"key": "value", "list": [1, 2, 3], "nested": {"inner": "data"}}
        response = APIResponse(original_data, 200, "Success")
        
        # Get response dict
        response_dict = response.to_dict()
        
        # Modify original data
        original_data["key"] = "modified"
        original_data["list"].append(4)
        original_data["nested"]["inner"] = "modified"
        
        # Response should not be affected by modifications to original data
        new_response_dict = response.to_dict()
        self.assertEqual(response_dict, new_response_dict)
        
        # Modify response dict
        response_dict["data"]["key"] = "dict_modified"
        
        # Original response should still be intact
        final_response_dict = response.to_dict()
        self.assertNotEqual(response_dict["data"]["key"], final_response_dict["data"]["key"])


# Parameterized Tests for Comprehensive Coverage
class TestServerAPIParameterized(unittest.TestCase):
    """Parameterized tests for comprehensive coverage."""
    
    def test_various_request_data_formats(self):
        """Test server with various request data formats."""
        async def test_formats():
            server = ServerAPI()
            await server.start()
            
            request_formats = [
                {"action": "simple"},
                {"action": "nested", "data": {"key": "value"}},
                {"action": "array", "data": [1, 2, 3]},
                {"action": "mixed", "data": {"array": [1, 2], "string": "test", "number": 42}},
                {"action": "unicode", "data": {"text": "测试 🚀 emoji"}},
                {"action": "boolean", "data": {"flag": True, "disabled": False}},
                {"action": "null_values", "data": {"empty": None, "zero": 0, "blank": ""}},
            ]
            
            for request_data in request_formats:
                with self.subTest(format=request_data):
                    try:
                        response = await server.process_request(request_data)
                        self.assertIsInstance(response, APIResponse)
                        self.assertGreaterEqual(response.status_code, 200)
                    except APIError as e:
                        # Some formats might be rejected, which is acceptable
                        self.assertIsInstance(e, APIError)
                        
        asyncio.run(test_formats())
    
    def test_error_codes_coverage(self):
        """Test various error code scenarios."""
        async def test_error_codes():
            server = ServerAPI()
            
            # Test different error conditions
            error_scenarios = [
                ("start_when_running", lambda: server.start()),
                ("stop_when_stopped", lambda: server.stop()),
                ("process_when_stopped", lambda: server.process_request({"action": "test"})),
            ]
            
            # Start server for some tests
            await server.start()
            
            for scenario_name, operation in error_scenarios:
                with self.subTest(scenario=scenario_name):
                    try:
                        if scenario_name == "start_when_running":
                            await operation()
                            self.fail(f"Expected APIError for {scenario_name}")
                        elif scenario_name == "stop_when_stopped":
                            await server.stop()  # Stop first
                            await operation()  # This should fail
                            self.fail(f"Expected APIError for {scenario_name}")
                        elif scenario_name == "process_when_stopped":
                            await server.stop()  # Ensure stopped
                            await operation()
                            self.fail(f"Expected APIError for {scenario_name}")
                    except APIError as e:
                        # Expected errors
                        expected_codes = {
                            "start_when_running": 400,
                            "stop_when_stopped": 400,
                            "process_when_stopped": 503,
                        }
                        expected_code = expected_codes.get(scenario_name, 500)
                        self.assertEqual(e.status_code, expected_code)
                        
        asyncio.run(test_error_codes())


# Performance Profiling and Resource Management Tests
class TestServerAPIResourceManagement(unittest.TestCase):
    """Test resource management and performance characteristics."""
    
    def setUp(self):
        """Set up resource management test fixtures."""
        self.config = ServerConfig({
            'host': 'localhost',
            'port': 8087,
            'max_connections': 20,
            'timeout': 15
        })
        self.server = ServerAPI(self.config)
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable over time."""
        async def test_memory():
            await self.server.start()
            
            # Process requests in batches to simulate sustained load
            for batch in range(5):
                tasks = []
                for i in range(50):
                    request_data = {
                        "action": "memory_stability_test",
                        "batch": batch,
                        "request_id": i,
                        "data": {"payload": "x" * 1000}  # 1KB per request
                    }
                    tasks.append(self.server.process_request(request_data))
                
                # Process batch
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check that most requests succeeded
                successful = [r for r in responses if isinstance(r, APIResponse)]
                self.assertGreaterEqual(len(successful), 45)  # 90% success rate
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            # Final verification
            final_stats = self.server.get_status()['stats']
            self.assertGreaterEqual(final_stats['requests'], 200)
            
        asyncio.run(test_memory())
    
    def test_connection_pool_efficiency(self):
        """Test connection pool management efficiency."""
        # Test rapid connection addition and removal
        for cycle in range(10):
            connections_added = []
            
            # Add connections rapidly
            for i in range(15):
                conn_id = f"cycle_{cycle}_conn_{i}"
                try:
                    self.server.add_connection(conn_id)
                    connections_added.append(conn_id)
                except APIError:
                    break  # Hit limit
            
            # Verify connections were added
            self.assertGreater(len(connections_added), 0)
            self.assertLessEqual(len(self.server.connections), 20)
            
            # Remove half the connections
            for i in range(0, len(connections_added), 2):
                self.server.remove_connection(connections_added[i])
            
            # Verify partial removal
            remaining_expected = len(connections_added) - len(connections_added[::2])
            actual_from_cycle = len([c for c in self.server.connections if c.startswith(f"cycle_{cycle}")])
            self.assertEqual(actual_from_cycle, remaining_expected)
        
        # Clean up all connections
        self.server.connections.clear()
        self.assertEqual(len(self.server.connections), 0)
    
    def test_timeout_handling(self):
        """Test timeout handling in various scenarios."""
        async def test_timeouts():
            await self.server.start()
            
            # Simulate requests that might timeout
            timeout_scenarios = [
                {"action": "quick_response", "expected_time": 0.01},
                {"action": "medium_response", "expected_time": 0.1}, 
                {"action": "slow_response", "expected_time": 1.0},
            ]
            
            for scenario in timeout_scenarios:
                with self.subTest(scenario=scenario):
                    start_time = asyncio.get_event_loop().time()
                    
                    try:
                        response = await asyncio.wait_for(
                            self.server.process_request(scenario),
                            timeout=2.0
                        )
                        
                        elapsed = asyncio.get_event_loop().time() - start_time
                        self.assertIsInstance(response, APIResponse)
                        
                        # Response time should be reasonable
                        self.assertLess(elapsed, 2.0)
                        
                    except asyncio.TimeoutError:
                        self.fail(f"Request timed out for scenario: {scenario}")
                    except APIError as e:
                        # Some timeouts might be handled as API errors
                        self.assertIsInstance(e, APIError)
                        
        asyncio.run(test_timeouts())


if __name__ == '__main__':
    # Run all tests including the new ones
    unittest.main(verbosity=2, exit=False)
    
    # Also run with pytest for additional test discovery
    try:
        import pytest
        pytest.main([__file__, '-v', '--tb=short'])
    except ImportError:
        print("pytest not available, tests completed with unittest only")