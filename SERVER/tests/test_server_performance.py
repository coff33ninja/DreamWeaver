"""
Performance and benchmarking tests for Server API.
Testing Framework: pytest with unittest compatibility
"""

import unittest
import pytest
import asyncio
import time
import statistics
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from test_server_api import ServerAPI, ServerConfig
except ImportError:
    # Import from the main test file
    pass


class TestServerAPIPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests for Server API."""

    def setUp(self):
        """Set up performance test environment."""
        self.config = ServerConfig(
            {"host": "localhost", "port": 8084, "max_connections": 1000, "timeout": 5}
        )
        self.server = ServerAPI(self.config)

    def tearDown(self):
        """Clean up performance test environment."""
        try:
            asyncio.run(self.server.stop())
        except:
            pass

    @pytest.mark.performance
    def test_server_startup_time(self):
        """Benchmark server startup time."""
        startup_times = []

        for _ in range(5):
            start_time = time.time()
            asyncio.run(self.server.start())
            end_time = time.time()

            startup_time = end_time - start_time
            startup_times.append(startup_time)

            asyncio.run(self.server.stop())

        avg_startup_time = statistics.mean(startup_times)
        max_startup_time = max(startup_times)

        # Assert reasonable startup times (adjust thresholds as needed)
        self.assertLess(avg_startup_time, 1.0, "Average startup time too slow")
        self.assertLess(max_startup_time, 2.0, "Maximum startup time too slow")

    @pytest.mark.performance
    def test_request_throughput(self):
        """Benchmark request processing throughput."""

        async def throughput_test():
            await self.server.start()

            num_requests = 100
            start_time = time.time()

            # Process multiple requests
            tasks = []
            for i in range(num_requests):
                request_data = {"action": "throughput_test", "id": i}
                task = self.server.process_request(request_data)
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            processing_time = end_time - start_time
            throughput = num_requests / processing_time

            # Check that most requests succeeded
            successful_responses = [
                r for r in responses if not isinstance(r, Exception)
            ]
            success_rate = len(successful_responses) / num_requests

            self.assertGreater(success_rate, 0.95, "Success rate too low")
            self.assertGreater(
                throughput, 50, "Throughput too low"
            )  # 50 requests/sec minimum

            return throughput

        throughput = asyncio.run(throughput_test())
        print(f"Request throughput: {throughput:.2f} requests/second")

    @pytest.mark.performance
    def test_memory_usage_stability(self):
        """Test memory usage stability over time."""
        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())

        async def memory_test():
            await self.server.start()

            initial_memory = process.memory_info().rss
            memory_readings = [initial_memory]

            # Process many requests to check for memory leaks
            for batch in range(10):
                tasks = []
                for i in range(20):
                    request_data = {"action": "memory_test", "batch": batch, "id": i}
                    task = self.server.process_request(request_data)
                    tasks.append(task)

                await asyncio.gather(*tasks, return_exceptions=True)

                # Force garbage collection
                gc.collect()

                # Record memory usage
                current_memory = process.memory_info().rss
                memory_readings.append(current_memory)

                # Small delay between batches
                await asyncio.sleep(0.1)

            # Check memory growth
            memory_growth = memory_readings[-1] - memory_readings[0]
            memory_growth_mb = memory_growth / (1024 * 1024)

            # Allow some memory growth but not excessive
            self.assertLess(
                memory_growth_mb,
                50,
                f"Excessive memory growth: {memory_growth_mb:.2f} MB",
            )

        asyncio.run(memory_test())

    @pytest.mark.performance
    def test_connection_handling_performance(self):
        """Test performance of connection management."""
        max_connections = self.config.get("max_connections", 1000)

        # Test adding many connections
        start_time = time.time()
        for i in range(max_connections):
            self.server.add_connection(f"perf_conn_{i}")
        add_time = time.time() - start_time

        self.assertEqual(len(self.server.connections), max_connections)

        # Test removing many connections
        start_time = time.time()
        for i in range(max_connections):
            self.server.remove_connection(f"perf_conn_{i}")
        remove_time = time.time() - start_time

        self.assertEqual(len(self.server.connections), 0)

        # Assert reasonable performance
        add_rate = max_connections / add_time
        remove_rate = max_connections / remove_time

        self.assertGreater(add_rate, 1000, "Connection addition rate too slow")
        self.assertGreater(remove_rate, 1000, "Connection removal rate too slow")

        print(f"Connection add rate: {add_rate:.0f} connections/second")
        print(f"Connection remove rate: {remove_rate:.0f} connections/second")


if __name__ == "__main__":
    unittest.main(verbosity=2)
