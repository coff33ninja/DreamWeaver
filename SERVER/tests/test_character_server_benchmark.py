import pytest
import time
import sys
import os
from statistics import mean, median

# Add the SERVER directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Testing framework: pytest
# Benchmark tests for character server performance


class TestCharacterServerBenchmarks:
    """Benchmark test suite for character server performance validation."""

    def test_character_creation_performance_benchmark(self):
        """Benchmark character creation performance."""
        from test_character_server import MockCharacterServer

        server = MockCharacterServer()
        char_data = {"name": "BenchmarkHero", "class": "Warrior", "level": 1}

        # Benchmark character creation
        creation_times = []
        num_iterations = 1000

        for i in range(num_iterations):
            char_data_with_id = {**char_data, "name": f"BenchmarkHero_{i}"}

            start_time = time.perf_counter()
            char = server.create_character(char_data_with_id)
            end_time = time.perf_counter()

            creation_times.append(end_time - start_time)
            assert char is not None

        # Analyze performance metrics
        avg_time = mean(creation_times)
        median_time = median(creation_times)
        max_time = max(creation_times)
        min_time = min(creation_times)

        # Performance assertions (adjust thresholds as needed)
        assert avg_time < 0.001, f"Average creation time too slow: {avg_time:.6f}s"
        assert max_time < 0.01, f"Maximum creation time too slow: {max_time:.6f}s"
        assert median_time < 0.001, f"Median creation time too slow: {median_time:.6f}s"

        print(f"Character Creation Benchmark Results:")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Median time: {median_time:.6f}s")
        print(f"  Min time: {min_time:.6f}s")
        print(f"  Max time: {max_time:.6f}s")

    def test_character_retrieval_performance_benchmark(self):
        """Benchmark character retrieval performance."""
        from test_character_server import MockCharacterServer

        server = MockCharacterServer()

        # Create characters for retrieval testing
        character_ids = []
        for i in range(1000):
            char_data = {
                "name": f"RetrievalHero_{i}",
                "class": "Warrior",
                "level": i % 100 + 1,
            }
            char = server.create_character(char_data)
            character_ids.append(char["id"])

        # Benchmark character retrieval
        retrieval_times = []
        num_retrievals = 1000

        import random

        for _ in range(num_retrievals):
            char_id = random.choice(character_ids)

            start_time = time.perf_counter()
            char = server.get_character(char_id)
            end_time = time.perf_counter()

            retrieval_times.append(end_time - start_time)
            assert char is not None

        # Analyze performance metrics
        avg_time = mean(retrieval_times)
        median_time = median(retrieval_times)
        max_time = max(retrieval_times)
        min_time = min(retrieval_times)

        # Performance assertions
        assert avg_time < 0.0005, f"Average retrieval time too slow: {avg_time:.6f}s"
        assert max_time < 0.005, f"Maximum retrieval time too slow: {max_time:.6f}s"

        print(f"Character Retrieval Benchmark Results:")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Median time: {median_time:.6f}s")
        print(f"  Min time: {min_time:.6f}s")
        print(f"  Max time: {max_time:.6f}s")

    def test_character_update_performance_benchmark(self):
        """Benchmark character update performance."""
        from test_character_server import MockCharacterServer

        server = MockCharacterServer()

        # Create characters for update testing
        characters = []
        for i in range(500):
            char_data = {"name": f"UpdateHero_{i}", "class": "Warrior", "level": 1}
            char = server.create_character(char_data)
            characters.append(char)

        # Benchmark character updates
        update_times = []
        num_updates = 1000

        import random

        for i in range(num_updates):
            char = random.choice(characters)
            update_data = {"level": random.randint(1, 100), "update_count": i}

            start_time = time.perf_counter()
            updated_char = server.update_character(char["id"], update_data)
            end_time = time.perf_counter()

            update_times.append(end_time - start_time)
            assert updated_char is not None

        # Analyze performance metrics
        avg_time = mean(update_times)
        median_time = median(update_times)
        max_time = max(update_times)
        min_time = min(update_times)

        # Performance assertions
        assert avg_time < 0.001, f"Average update time too slow: {avg_time:.6f}s"
        assert max_time < 0.01, f"Maximum update time too slow: {max_time:.6f}s"

        print(f"Character Update Benchmark Results:")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Median time: {median_time:.6f}s")
        print(f"  Min time: {min_time:.6f}s")
        print(f"  Max time: {max_time:.6f}s")

    def test_character_list_performance_scaling_benchmark(self):
        """Benchmark character listing performance with different dataset sizes."""
        from test_character_server import MockCharacterServer

        server = MockCharacterServer()

        dataset_sizes = [100, 500, 1000, 2500, 5000]
        list_times = []

        for size in dataset_sizes:
            # Create characters up to the target size
            current_count = len(server.list_characters())
            while current_count < size:
                char_data = {
                    "name": f"ListHero_{current_count}",
                    "class": "Warrior",
                    "level": 1,
                }
                server.create_character(char_data)
                current_count += 1

            # Benchmark list operation
            start_time = time.perf_counter()
            all_chars = server.list_characters()
            end_time = time.perf_counter()

            list_time = end_time - start_time
            list_times.append((size, list_time))

            assert len(all_chars) == size
            print(f"List {size} characters: {list_time:.6f}s")

        # Analyze scaling behavior
        for i in range(1, len(list_times)):
            prev_size, prev_time = list_times[i - 1]
            curr_size, curr_time = list_times[i]

            # Calculate scaling factor
            size_ratio = curr_size / prev_size
            time_ratio = curr_time / prev_time if prev_time > 0 else float("inf")

            # Should not scale worse than linearly (with some tolerance)
            assert (
                time_ratio < size_ratio * 2
            ), f"List performance scaling too poor: {time_ratio:.2f}x time for {size_ratio:.2f}x data"

    def test_mixed_operations_performance_benchmark(self):
        """Benchmark mixed character operations performance."""
        from test_character_server import MockCharacterServer

        server = MockCharacterServer()

        # Pre-populate with some characters
        for i in range(1000):
            char_data = {"name": f"MixedHero_{i}", "class": "Warrior", "level": 1}
            server.create_character(char_data)

        # Benchmark mixed operations
        operation_times = []
        num_operations = 2000

        import random

        operations = ["create", "read", "update", "delete", "list"]

        for i in range(num_operations):
            operation = random.choice(operations)

            start_time = time.perf_counter()

            if operation == "create":
                char_data = {"name": f"NewHero_{i}", "class": "Warrior", "level": 1}
                server.create_character(char_data)

            elif operation == "read":
                all_chars = server.list_characters()
                if all_chars:
                    char = random.choice(all_chars)
                    server.get_character(char["id"])

            elif operation == "update":
                all_chars = server.list_characters()
                if all_chars:
                    char = random.choice(all_chars)
                    server.update_character(
                        char["id"], {"level": random.randint(1, 100)}
                    )

            elif operation == "delete":
                all_chars = server.list_characters()
                if len(all_chars) > 500:  # Keep minimum number of characters
                    char = random.choice(all_chars)
                    server.delete_character(char["id"])

            elif operation == "list":
                server.list_characters()

            end_time = time.perf_counter()
            operation_times.append((operation, end_time - start_time))

        # Analyze performance by operation type
        operation_stats = {}
        for op_type in operations:
            times = [time for op, time in operation_times if op == op_type]
            if times:
                operation_stats[op_type] = {
                    "count": len(times),
                    "avg_time": mean(times),
                    "max_time": max(times),
                    "total_time": sum(times),
                }

        # Performance assertions for each operation type
        for op_type, stats in operation_stats.items():
            assert (
                stats["avg_time"] < 0.01
            ), f"{op_type} average time too slow: {stats['avg_time']:.6f}s"
            assert (
                stats["max_time"] < 0.1
            ), f"{op_type} maximum time too slow: {stats['max_time']:.6f}s"

            print(f"{op_type.upper()} Operation Benchmark:")
            print(f"  Count: {stats['count']}")
            print(f"  Average time: {stats['avg_time']:.6f}s")
            print(f"  Max time: {stats['max_time']:.6f}s")
            print(f"  Total time: {stats['total_time']:.6f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])  # -s to show print statements
