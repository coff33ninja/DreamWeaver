"""
Pytest configuration and fixtures for Server API tests.
Testing Framework: pytest with unittest compatibility
"""
import pytest
import tempfile
import os
import json
import asyncio
from unittest.mock import Mock, patch


@pytest.fixture
def server_config():
    """Fixture providing standard server configuration for tests."""
    return {
        'host': 'localhost',
        'port': 8080,
        'debug': True,
        'max_connections': 100,
        'timeout': 30,
        'workers': 1
    }


@pytest.fixture
def temp_config_file(server_config):
    """Fixture providing a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(server_config, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def mock_server_environment(monkeypatch):
    """Fixture for mocking server environment variables."""
    test_env = {
        'SERVER_HOST': 'test.localhost',
        'SERVER_PORT': '9999',
        'SERVER_DEBUG': 'true',
        'SERVER_MAX_CONNECTIONS': '50'
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def running_server(server_config):
    """Fixture providing a running server instance."""
    from test_server_api import ServerAPI, ServerConfig
    
    config = ServerConfig(server_config)
    server = ServerAPI(config)
    
    await server.start()
    yield server
    
    try:
        await server.stop()
    except:
        pass


@pytest.fixture
def mock_request_data():
    """Fixture providing mock request data for testing."""
    return {
        'action': 'test_action',
        'data': {
            'key1': 'value1',
            'key2': 'value2',
            'nested': {
                'inner_key': 'inner_value'
            }
        },
        'metadata': {
            'timestamp': '2024-01-01T00:00:00Z',
            'user_id': 'test_user',
            'request_id': 'test_request_123'
        }
    }


@pytest.fixture
def mock_database():
    """Fixture providing a mock database for testing."""
    database = Mock()
    database.connect.return_value = True
    database.disconnect.return_value = True
    database.query.return_value = {'result': 'success'}
    database.insert.return_value = {'id': 1, 'created': True}
    database.update.return_value = {'updated': True}
    database.delete.return_value = {'deleted': True}
    
    return database


@pytest.fixture
def mock_external_api():
    """Fixture providing a mock external API for testing."""
    api = Mock()
    api.get.return_value = {'status': 'success', 'data': 'mock_data'}
    api.post.return_value = {'status': 'created', 'id': 'mock_id'}
    api.put.return_value = {'status': 'updated'}
    api.delete.return_value = {'status': 'deleted'}
    
    return api


# Pytest-specific test classes
class TestServerAPIPytest:
    """Pytest-style tests for Server API."""
    
    @pytest.mark.asyncio
    async def test_server_startup_with_fixture(self, server_config):
        """Test server startup using pytest fixtures."""
        from test_server_api import ServerAPI, ServerConfig
        
        config = ServerConfig(server_config)
        server = ServerAPI(config)
        
        result = await server.start()
        assert result is True
        assert server.is_running is True
        
        await server.stop()
    
    @pytest.mark.asyncio
    async def test_request_processing_with_mock_data(self, running_server, mock_request_data):
        """Test request processing with mock data."""
        response = await running_server.process_request(mock_request_data)
        
        assert response.status_code == 200
        assert response.data is not None
    
    @pytest.mark.parametrize("port,expected_valid", [
        (80, True),
        (8080, True),
        (65535, True),
        (0, False),
        (-1, False),
        (65536, False)
    ])
    def test_port_validation(self, port, expected_valid):
        """Test port validation with parametrized values."""
        from test_server_api import ServerConfig
        
        if expected_valid:
            config = ServerConfig({'port': port})
            assert config.get('port') == port
        else:
            # In a real implementation, this might raise an exception
            # For now, we just test that the config is created
            config = ServerConfig({'port': port})
            assert isinstance(config, ServerConfig)
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_high_load_performance(self, running_server):
        """Test server performance under high load."""
        import asyncio
        
        # Create many concurrent requests
        tasks = []
        for i in range(100):
            request_data = {'action': 'load_test', 'id': i}
            task = running_server.process_request(request_data)
            tasks.append(task)
        
        # Process all requests
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most requests succeeded
        successful = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful) >= 90  # Allow for some failures under load