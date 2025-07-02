"""
Pytest configuration for character server tests.
Provides additional fixtures and configuration for comprehensive testing.
"""

import pytest
import time
import random
from typing import Dict, List, Any


@pytest.fixture(scope="session")
def test_configuration():
    """Session-scoped fixture providing test configuration."""
    return {
        'max_test_characters': 10000,
        'performance_threshold_ms': 10,
        'stress_test_iterations': 1000,
        'concurrent_operation_batches': 5,
        'large_dataset_size': 500,
    }


@pytest.fixture
def performance_monitor():
    """Fixture for monitoring test performance."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.measurements = []
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self, operation_name: str):
            if self.start_time:
                duration = time.time() - self.start_time
                self.measurements.append({
                    'operation': operation_name,
                    'duration_ms': duration * 1000
                })
                self.start_time = None
                return duration
        
        def get_average_duration(self) -> float:
            if not self.measurements:
                return 0.0
            return sum(m['duration_ms'] for m in self.measurements) / len(self.measurements)
    
    return PerformanceMonitor()


@pytest.fixture
def stress_test_data_generator():
    """Fixture for generating stress test data."""
    def generate_character_data(count: int, prefix: str = "StressTest") -> List[Dict[str, Any]]:
        characters = []
        for i in range(count):
            characters.append({
                'name': f'{prefix}Hero_{i}',
                'class': random.choice(['Warrior', 'Mage', 'Rogue', 'Cleric', 'Paladin']),
                'level': random.randint(1, 100),
                'hp': random.randint(50, 500),
                'mp': random.randint(20, 300),
                'strength': random.randint(5, 25),
                'dexterity': random.randint(5, 25),
                'intelligence': random.randint(5, 25),
                'equipment': [f'item_{j}' for j in range(random.randint(1, 10))],
                'skills': [f'skill_{j}' for j in range(random.randint(1, 5))],
            })
        return characters
    
    return generate_character_data


@pytest.fixture
def character_server_populated(character_server, stress_test_data_generator):
    """Fixture providing a character server pre-populated with test data."""
    # Generate and create test characters
    test_characters = stress_test_data_generator(50, "PrePopulated")
    created_characters = []
    
    for char_data in test_characters:
        char = character_server.create_character(char_data)
        created_characters.append(char)
    
    # Attach the created characters to the server for reference
    character_server._test_characters = created_characters
    
    return character_server


# Custom pytest markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (may take several seconds)")
    config.addinivalue_line("markers", "stress: marks tests as stress tests (high resource usage)")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "security: marks tests as security-related tests")
    config.addinivalue_line("markers", "compatibility: marks tests as compatibility tests")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify collected test items to add markers based on test names."""
    for item in items:
        # Mark performance tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Mark stress tests
        if "stress" in item.name.lower() or "large" in item.name.lower():
            item.add_marker(pytest.mark.stress)
            item.add_marker(pytest.mark.slow)
        
        # Mark security tests
        if "security" in item.name.lower() or "injection" in item.name.lower():
            item.add_marker(pytest.mark.security)
        
        # Mark compatibility tests
        if "compatibility" in item.name.lower() or "interoperability" in item.name.lower():
            item.add_marker(pytest.mark.compatibility)