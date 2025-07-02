"""
Pytest configuration for enhanced ServerAPI tests.
"""
import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def server_config():
    """Provide test server configuration."""
    from SERVER.src.server_config import ServerConfig
    return ServerConfig({
        'host': 'localhost',
        'port': 8080,
        'debug': True,
        'max_connections': 10,
        'timeout': 30
    })


# Custom markers for test categorization
pytest_plugins = []

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "security: Security-related tests")
    config.addinivalue_line("markers", "edge_case: Edge case tests")
    config.addinivalue_line("markers", "data_integrity: Data integrity tests")
    config.addinivalue_line("markers", "performance: Performance tests")