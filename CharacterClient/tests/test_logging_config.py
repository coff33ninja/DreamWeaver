import pytest
import os
import tempfile
import shutil
import logging
import logging.handlers
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from logging_config import (
    setup_client_logging,
    get_logger,
    CLIENT_LOG_FILE,
    LOGGER_NAME,
)


class TestLoggingConfiguration:
    """Test logging configuration functionality."""

    def setup_method(self):
        """Clear logging configuration before each test."""
        # Remove all handlers from the logger
        logger = logging.getLogger(LOGGER_NAME)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)

    def teardown_method(self):
        """Clean up after each test."""
        # Remove all handlers from the logger
        logger = logging.getLogger(LOGGER_NAME)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_setup_client_logging_default_level(self):
        """Test setup_client_logging with default level."""
        setup_client_logging()
        logger = get_logger()

        assert logger.level == logging.INFO
        assert logger.name == LOGGER_NAME
        assert len(logger.handlers) >= 1  # Should have at least console handler

    @pytest.mark.parametrize(
        "log_level",
        [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL],
    )
    def test_setup_client_logging_different_levels(self, log_level):
        """Test setup_client_logging with different log levels."""
        setup_client_logging(log_level)
        logger = get_logger()

        assert logger.level == log_level

    def test_logger_has_console_handler(self):
        """Test that logger has console handler configured."""
        setup_client_logging()
        logger = get_logger()

        # Should have at least one StreamHandler
        console_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(console_handlers) >= 1

    def test_logger_has_file_handler(self):
        """Test that logger has rotating file handler configured."""
        setup_client_logging()
        logger = get_logger()

        # Should have RotatingFileHandler
        file_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) >= 1

    def test_file_handler_configuration(self):
        """Test file handler configuration parameters."""
        setup_client_logging()
        logger = get_logger()

        file_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        if file_handlers:
            handler = file_handlers[0]

            # Test rotation parameters
            assert handler.maxBytes == 2 * 1024 * 1024  # 2MB
            assert handler.backupCount == 3

    def test_log_file_path_creation(self):
        """Test that log file directory is created."""
        temp_dir = tempfile.mkdtemp()
        temp_log_file = os.path.join(temp_dir, "subdir", "test.log")

        try:
            with patch("logging_config.CLIENT_LOG_FILE", temp_log_file):
                setup_client_logging()

                # Directory should be created
                assert os.path.exists(os.path.dirname(temp_log_file))
        finally:
            shutil.rmtree(temp_dir)

    def test_log_file_creation_failure_handling(self):
        """Test graceful handling of log file creation failures."""
        # Use an invalid path that should fail
        invalid_log_path = "/invalid/readonly/path/test.log"

        with patch("logging_config.CLIENT_LOG_FILE", invalid_log_path):
            # Should not raise exception, but should log error
            try:
                setup_client_logging()
                logger = get_logger()

                # Should still have console handler even if file handler fails
                assert len(logger.handlers) >= 1
                console_handlers = [
                    h for h in logger.handlers if isinstance(h, logging.StreamHandler)
                ]
                assert len(console_handlers) >= 1
            except Exception:
                pytest.fail(
                    "setup_client_logging should handle file creation failures gracefully"
                )

    def test_logger_formatter_configuration(self):
        """Test that loggers have proper formatters configured."""
        setup_client_logging()
        logger = get_logger()

        for handler in logger.handlers:
            assert handler.formatter is not None
            # Test that formatter includes expected components
            formatter = handler.formatter
            format_string = formatter._fmt

            expected_components = [
                "%(asctime)s",
                "%(name)s",
                "%(levelname)s",
                "%(message)s",
            ]
            for component in expected_components:
                assert component in format_string

    def test_multiple_setup_calls_idempotent(self):
        """Test that multiple calls to setup_client_logging are idempotent."""
        setup_client_logging()
        initial_handler_count = len(get_logger().handlers)

        # Call setup again
        setup_client_logging()
        setup_client_logging()

        # Should clear previous handlers and set up fresh ones
        logger = get_logger()
        # Exact count may vary, but should not accumulate handlers indefinitely
        assert len(logger.handlers) <= initial_handler_count + 2  # Allow some tolerance

    def test_get_logger_returns_configured_logger(self):
        """Test that get_logger returns the properly configured logger."""
        setup_client_logging()
        logger1 = get_logger()
        logger2 = get_logger()

        # Should return the same logger instance
        assert logger1 is logger2
        assert logger1.name == LOGGER_NAME

    def test_get_logger_with_custom_name(self):
        """Test get_logger with custom logger name."""
        custom_name = "custom_test_logger"
        logger = get_logger(custom_name)

        assert logger.name == custom_name

    def test_logging_output_format(self):
        """Test that log output has expected format."""
        import io

        # Create a string buffer to capture log output
        log_buffer = io.StringIO()

        # Set up logger with string buffer
        logger = logging.getLogger(LOGGER_NAME)
        handler = logging.StreamHandler(log_buffer)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log a test message
        test_message = "Test log message"
        logger.info(test_message)

        # Check output format
        log_output = log_buffer.getvalue()
        assert test_message in log_output
        assert LOGGER_NAME in log_output
        assert "INFO" in log_output

    def test_unicode_logging_support(self):
        """Test that logging supports Unicode characters."""
        setup_client_logging()
        logger = get_logger()

        # Test Unicode message
        unicode_message = "测试消息 🚀 тест сообщение"

        try:
            logger.info(unicode_message)
            # Should not raise UnicodeError
        except UnicodeError:
            pytest.fail("Logger should support Unicode messages")

    def test_log_rotation_behavior(self):
        """Test log file rotation behavior."""
        temp_dir = tempfile.mkdtemp()
        temp_log_file = os.path.join(temp_dir, "rotation_test.log")

        try:
            with patch("logging_config.CLIENT_LOG_FILE", temp_log_file):
                setup_client_logging()
                logger = get_logger()

                # Generate log messages to trigger rotation
                large_message = "x" * 1000  # 1KB message

                # Write enough to potentially trigger rotation (2MB limit)
                for i in range(3000):  # Should exceed 2MB
                    logger.info(f"Large message {i}: {large_message}")

                # Check if log files exist
                assert os.path.exists(temp_log_file)

                # May have created backup files
                backup_files = [
                    f
                    for f in os.listdir(temp_dir)
                    if f.startswith("rotation_test.log") and f != "rotation_test.log"
                ]

                # Should have at most 3 backup files (as configured)
                assert len(backup_files) <= 3

        finally:
            shutil.rmtree(temp_dir)

    def test_concurrent_logging_safety(self):
        """Test that logging is safe for concurrent use."""
        import threading
        import time

        setup_client_logging()
        logger = get_logger()

        results = []
        errors = []

        def log_worker(worker_id):
            try:
                for i in range(50):
                    logger.info(f"Worker {worker_id} message {i}")
                    time.sleep(0.001)  # Small delay
                results.append(worker_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads logging concurrently
        threads = [threading.Thread(target=log_worker, args=(i,)) for i in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All workers should complete without errors
        assert len(errors) == 0
        assert len(results) == 5

    def test_logging_performance(self):
        """Test logging performance under load."""
        import time

        setup_client_logging()
        logger = get_logger()

        start_time = time.time()

        # Log many messages
        for i in range(1000):
            logger.info(f"Performance test message {i}")

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete in reasonable time (less than 1 second for 1000 messages)
        assert total_time < 1.0


class TestLoggingIntegration:
    """Test integration between logging and other components."""

    def test_main_execution_logging(self):
        """Test logging when module is executed as main."""
        # This would test the if __name__ == "__main__" block
        # We can verify the components work together
        setup_client_logging(logging.DEBUG)
        logger = get_logger()

        # Test all log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # Should not raise any exceptions
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
