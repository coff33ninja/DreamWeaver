import pytest
import os
import tempfile
import shutil
from unittest.mock import patch
import sys

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from logging_config import setup_client_logging, get_logger, CLIENT_LOG_FILE


class TestConfigLoggingIntegration:
    """Test integration between config and logging modules."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.environ.clear()
        os.environ.update(self.original_env)

        # Clean up logging
        logger = get_logger()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_logging_uses_config_paths(self):
        """Test that logging configuration uses paths from config module."""
        # Set custom data path
        custom_data_path = os.path.join(self.temp_dir, "custom_client_data")

        with patch.dict(os.environ, {"DREAMWEAVER_CLIENT_DATA_PATH": custom_data_path}):
            import importlib

            importlib.reload(config)

            # Import logging_config after config is reloaded
            if "logging_config" in sys.modules:
                importlib.reload(sys.modules["logging_config"])
            else:
                import logging_config

            # Log file should be in custom path
            assert logging_config.CLIENT_LOG_FILE.startswith(custom_data_path)

    def test_end_to_end_directory_and_logging_setup(self):
        """Test complete setup flow from config to logging."""
        custom_base = os.path.join(self.temp_dir, "e2e_test")

        with patch.dict(os.environ, {"DREAMWEAVER_CLIENT_DATA_PATH": custom_base}):
            # Reload config to pick up new environment
            import importlib

            importlib.reload(config)

            # Reload logging config
            if "logging_config" in sys.modules:
                importlib.reload(sys.modules["logging_config"])

            from logging_config import setup_client_logging, get_logger

            # Set up logging
            setup_client_logging()
            logger = get_logger()

            # Test that everything works together
            logger.info("End-to-end test message")

            # Verify directory structure was created
            assert os.path.exists(custom_base)
            assert os.path.exists(os.path.join(custom_base, "logs"))

            # Verify log file is in correct location
            expected_log_file = os.path.join(custom_base, "logs", "client.log")
            assert os.path.exists(expected_log_file)

    def test_config_changes_affect_logging(self):
        """Test that changes to config affect logging behavior."""
        # Set initial path
        initial_path = os.path.join(self.temp_dir, "initial")

        with patch.dict(os.environ, {"DREAMWEAVER_CLIENT_DATA_PATH": initial_path}):
            import importlib

            importlib.reload(config)

            if "logging_config" in sys.modules:
                importlib.reload(sys.modules["logging_config"])

            from logging_config import CLIENT_LOG_FILE as initial_log_file

            assert initial_log_file.startswith(initial_path)

            # Change path
            new_path = os.path.join(self.temp_dir, "changed")
            with patch.dict(os.environ, {"DREAMWEAVER_CLIENT_DATA_PATH": new_path}):
                importlib.reload(config)
                importlib.reload(sys.modules["logging_config"])

                from logging_config import CLIENT_LOG_FILE as new_log_file

                assert new_log_file.startswith(new_path)
                assert new_log_file != initial_log_file

    def test_missing_log_directory_creation(self):
        """Test that missing log directory is created when needed."""
        custom_base = os.path.join(self.temp_dir, "missing_dir_test")

        # Don't create the directory initially
        with patch.dict(os.environ, {"DREAMWEAVER_CLIENT_DATA_PATH": custom_base}):
            import importlib

            importlib.reload(config)

            # Directory should be created by config module
            assert os.path.exists(custom_base)
            assert os.path.exists(os.path.join(custom_base, "logs"))

            # Set up logging
            if "logging_config" in sys.modules:
                importlib.reload(sys.modules["logging_config"])

            from logging_config import setup_client_logging, get_logger

            setup_client_logging()

            # Should work without errors
            logger = get_logger()
            logger.info("Test message in created directory")

    def test_config_main_execution_with_logging(self):
        """Test that config main execution works with logging setup."""
        # This tests the interaction between the modules when config is run as main
        custom_base = os.path.join(self.temp_dir, "main_exec_test")

        with patch.dict(os.environ, {"DREAMWEAVER_CLIENT_DATA_PATH": custom_base}):
            import importlib

            importlib.reload(config)

            # All directories should be created
            assert os.path.exists(config.CLIENT_DATA_PATH)
            assert os.path.exists(config.CLIENT_LOGS_PATH)

            # Logging should be able to use these paths
            if "logging_config" in sys.modules:
                importlib.reload(sys.modules["logging_config"])

            from logging_config import setup_client_logging

            setup_client_logging()

            # Should work without errors
            assert True


class TestConfigEnvironmentPrecedence:
    """Test environment variable precedence and inheritance."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_data_path_override_affects_derived_paths(self):
        """Test that overriding data path affects all derived paths."""
        custom_data_path = os.path.join(self.temp_dir, "custom_data")

        with patch.dict(os.environ, {"DREAMWEAVER_CLIENT_DATA_PATH": custom_data_path}):
            import importlib

            importlib.reload(config)

            # All derived paths should use the custom base
            assert config.CLIENT_DATA_PATH == custom_data_path
            assert config.CLIENT_LOGS_PATH.startswith(custom_data_path)
            assert config.CLIENT_TEMP_AUDIO_PATH.startswith(custom_data_path)

            # Models path should also be affected unless separately overridden
            if not os.environ.get("DREAMWEAVER_CLIENT_MODELS_PATH"):
                assert config.CLIENT_MODELS_PATH.startswith(custom_data_path)

    def test_independent_models_path_override(self):
        """Test that models path can be overridden independently."""
        custom_data_path = os.path.join(self.temp_dir, "data")
        custom_models_path = os.path.join(self.temp_dir, "separate_models")

        with patch.dict(
            os.environ,
            {
                "DREAMWEAVER_CLIENT_DATA_PATH": custom_data_path,
                "DREAMWEAVER_CLIENT_MODELS_PATH": custom_models_path,
            },
        ):
            import importlib

            importlib.reload(config)

            # Data path should use custom data path
            assert config.CLIENT_DATA_PATH == custom_data_path

            # Models path should use separate custom path
            assert config.CLIENT_MODELS_PATH == custom_models_path

            # Other derived paths should still use data path
            assert config.CLIENT_LOGS_PATH.startswith(custom_data_path)
            assert config.CLIENT_TEMP_AUDIO_PATH.startswith(custom_data_path)

    @pytest.mark.parametrize(
        "env_var,expected_attr",
        [
            ("DREAMWEAVER_CLIENT_DATA_PATH", "CLIENT_DATA_PATH"),
            ("DREAMWEAVER_CLIENT_MODELS_PATH", "CLIENT_MODELS_PATH"),
        ],
    )
    def test_environment_variable_precedence(self, env_var, expected_attr):
        """Test that environment variables take precedence over defaults."""
        custom_path = os.path.join(self.temp_dir, f"custom_{env_var.lower()}")

        with patch.dict(os.environ, {env_var: custom_path}):
            import importlib

            importlib.reload(config)

            actual_value = getattr(config, expected_attr)
            assert actual_value == custom_path


if __name__ == "__main__":
    pytest.main([__file__])
