"""
Comprehensive performance benchmarks for the character server.
Testing framework: pytest with pytest-benchmark
"""

import pytest
import time
import sys
import os
from unittest.mock import Mock, patch
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the SERVER directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the mock server for benchmarking
from test_character_server import MockCharacterServer


class TestCharacterServerBenchmarks:
    """Benchmark suite for character server performance testing."""
    
    @pytest.fixture
    def benchmark_server(self):
        """Fixture providing a fresh server for benchmark tests."""
        return MockCharacterServer()
    
    @pytest.fixture
    def benchmark_character_data(self):
        """Standard character data for benchmarking."""
        return {
            'name': 'BenchmarkHero',
            'class': 'Speedster',
            'level': 50,
            'hp': 500,
            'mp': 250,
            'experience': 25000,
            'skills': ['speed', 'agility', 'quick_strike'],
            'equipment': ['speed_boots', 'light_armor', 'swift_blade']
        }
    
    def test_benchmark_character_creation(self, benchmark, benchmark_server, benchmark_character_data):
        """Benchmark character creation performance."""
        def create_character():
            return benchmark_server.create_character(benchmark_character_data)
        
        result = benchmark(create_character)
        assert result is not None
        assert 'id' in result
    
    def test_benchmark_character_retrieval(self, benchmark, benchmark_server, benchmark_character_data):
        """Benchmark character retrieval performance."""
        # Setup: create a character first
        char = benchmark_server.create_character(benchmark_character_data)
        
        def get_character():
            return benchmark_server.get_character(char['id'])
        
        result = benchmark(get_character)
        assert result is not None
        assert result['id'] == char['id']
    
    def test_benchmark_character_update(self, benchmark, benchmark_server, benchmark_character_data):
        """Benchmark character update performance."""
        # Setup: create a character first
        char = benchmark_server.create_character(benchmark_character_data)
        update_data = {'level': 75, 'hp': 750, 'experience': 50000}
        
        def update_character():
            return benchmark_server.update_character(char['id'], update_data)
        
        result = benchmark(update_character)
        assert result is not None
        assert result['level'] == 75
    
    def test_benchmark_character_deletion(self, benchmark, benchmark_server, benchmark_character_data):
        """Benchmark character deletion performance."""
        def delete_character():
            # Create character and delete it in the benchmark
            char = benchmark_server.create_character(benchmark_character_data)
            return benchmark_server.delete_character(char['id'])
        
        result = benchmark(delete_character)
        assert result is True
    
    def test_benchmark_character_listing(self, benchmark, benchmark_server, benchmark_character_data):
        """Benchmark character listing performance with pre-loaded data."""
        # Setup: create multiple characters
        for i in range(100):
            char_data = {**benchmark_character_data, 'name': f'ListHero_{i}'}
            benchmark_server.create_character(char_data)
        
        def list_characters():
            return benchmark_server.list_characters()
        
        result = benchmark(list_characters)
        assert len(result) == 100
    
    def test_benchmark_character_validation(self, benchmark, benchmark_character_data):
        """Benchmark character data validation performance."""
        def validate_character():
            return MockCharacterServer.validate_character_data(benchmark_character_data)
        
        result = benchmark(validate_character)
        assert result is True
    
    def test_benchmark_bulk_character_operations(self, benchmark, benchmark_server, benchmark_character_data):
        """Benchmark bulk character operations."""
        def bulk_operations():
            created_chars = []
            # Bulk create
            for i in range(50):
                char_data = {**benchmark_character_data, 'name': f'BulkHero_{i}'}
                char = benchmark_server.create_character(char_data)
                created_chars.append(char)
            
            # Bulk update
            for char in created_chars:
                benchmark_server.update_character(char['id'], {'level': 99})
            
            # Bulk retrieve
            for char in created_chars:
                benchmark_server.get_character(char['id'])
            
            # Bulk delete
            for char in created_chars:
                benchmark_server.delete_character(char['id'])
            
            return len(created_chars)
        
        result = benchmark(bulk_operations)
        assert result == 50


class TestCharacterServerMemoryBenchmarks:
    """Memory usage benchmarks for character server."""
    
    def test_memory_usage_character_creation(self, benchmark_character_data):
        """Test memory usage during character creation."""
        import psutil
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        benchmark_server = MockCharacterServer()
        
        # Create many characters and measure memory growth
        for i in range(1000):
            char_data = {**benchmark_character_data, 'name': f'MemHero_{i}'}
            benchmark_server.create_character(char_data)
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / (1024 * 1024)
        
        # Memory growth should be reasonable (less than 100MB for 1000 characters)
        assert memory_growth_mb < 100, f"Memory growth too high: {memory_growth_mb:.2f}MB"
    
    def test_memory_cleanup_after_character_deletion(self, benchmark_character_data):
        """Test memory cleanup after character deletion."""
        import psutil
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        benchmark_server = MockCharacterServer()
        
        # Create and delete characters
        for i in range(1000):
            char_data = {**benchmark_character_data, 'name': f'TempHero_{i}'}
            char = benchmark_server.create_character(char_data)
            benchmark_server.delete_character(char['id'])
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_difference = final_memory - initial_memory
        memory_difference_mb = memory_difference / (1024 * 1024)
        
        # Memory should not grow significantly after cleanup
        assert memory_difference_mb < 10, f"Memory not properly cleaned up: {memory_difference_mb:.2f}MB retained"


class TestCharacterServerConcurrencyBenchmarks:
    """Concurrency and threading benchmarks for character server."""
    
    def test_concurrent_character_creation_benchmark(self, benchmark_character_data):
        """Benchmark concurrent character creation."""
        server = MockCharacterServer()
        num_threads = 10
        operations_per_thread = 100
        
        def create_characters_batch(thread_id):
            results = []
            for i in range(operations_per_thread):
                char_data = {**benchmark_character_data, 'name': f'ConcurrentHero_{thread_id}_{i}'}
                char = server.create_character(char_data)
                results.append(char)
            return results
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_characters_batch, i) for i in range(num_threads)]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        end_time = time.time()
        total_time = end_time - start_time
        total_operations = num_threads * operations_per_thread
        operations_per_second = total_operations / total_time
        
        # Verify all operations completed successfully
        assert len(all_results) == total_operations
        
        # Performance should be reasonable (at least 100 operations per second)
        assert operations_per_second > 100, f"Concurrent performance too low: {operations_per_second:.2f} ops/sec"
    
    def test_mixed_concurrent_operations_benchmark(self, benchmark_character_data):
        """Benchmark mixed concurrent operations (CRUD)."""
        server = MockCharacterServer()
        
        # Pre-create some characters for read/update/delete operations
        initial_chars = []
        for i in range(50):
            char_data = {**benchmark_character_data, 'name': f'InitialHero_{i}'}
            char = server.create_character(char_data)
            initial_chars.append(char)
        
        def mixed_operations(thread_id):
            operations_count = {'create': 0, 'read': 0, 'update': 0, 'delete': 0, 'list': 0}
            
            for i in range(100):
                operation_type = i % 5
                
                if operation_type == 0:  # Create
                    char_data = {**benchmark_character_data, 'name': f'MixedHero_{thread_id}_{i}'}
                    server.create_character(char_data)
                    operations_count['create'] += 1
                
                elif operation_type == 1 and initial_chars:  # Read
                    char_id = initial_chars[i % len(initial_chars)]['id']
                    server.get_character(char_id)
                    operations_count['read'] += 1
                
                elif operation_type == 2 and initial_chars:  # Update
                    char_id = initial_chars[i % len(initial_chars)]['id']
                    server.update_character(char_id, {'level': 50 + i})
                    operations_count['update'] += 1
                
                elif operation_type == 3:  # List
                    server.list_characters()
                    operations_count['list'] += 1
                
                # Skip delete operations to maintain character pool
            
            return operations_count
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(5)]
            all_operations = []
            for future in as_completed(futures):
                all_operations.append(future.result())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate total operations performed
        total_ops = sum(sum(ops.values()) for ops in all_operations)
        ops_per_second = total_ops / total_time
        
        # Verify reasonable performance under mixed load
        assert ops_per_second > 500, f"Mixed operations performance too low: {ops_per_second:.2f} ops/sec"


if __name__ == '__main__':
    # Run benchmark tests
    try:
        pytest.main([
            __file__,
            '--benchmark-only',
            '--benchmark-sort=mean',
            '--benchmark-json=character_server_benchmarks.json',
            '--benchmark-histogram=histogram',
            '-v'
        ])
    except SystemExit:
        # If pytest-benchmark is not available, run regular tests
        pytest.main([__file__, '-v'])