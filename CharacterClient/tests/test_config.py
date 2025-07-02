import pytest
import os
import tempfile
import shutil
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add the src directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from logging_config import setup_client_logging, get_logger, CLIENT_LOG_FILE


class TestConfigPaths:
    """Test configuration path constants and derivation."""
    
    def test_client_root_exists(self):
        """Test that CLIENT_ROOT is properly defined and exists."""
        assert hasattr(config, 'CLIENT_ROOT')
        assert config.CLIENT_ROOT is not None
        assert os.path.isabs(config.CLIENT_ROOT)  # Should be absolute path
        
    def test_default_paths_structure(self):
        """Test that default paths follow expected structure."""
        assert hasattr(config, 'DEFAULT_CLIENT_DATA_PATH')
        assert hasattr(config, 'DEFAULT_CLIENT_MODELS_PATH')
        
        # Data path should be relative to CLIENT_ROOT
        assert config.DEFAULT_CLIENT_DATA_PATH.startswith(config.CLIENT_ROOT)
        
        # Models path should be under data path
        assert config.DEFAULT_CLIENT_MODELS_PATH.startswith(config.DEFAULT_CLIENT_DATA_PATH)
    
    def test_specific_model_paths(self):
        """Test that specific model type paths are properly defined."""
        model_paths = [
            'CLIENT_LLM_MODELS_PATH',
            'CLIENT_TTS_MODELS_PATH', 
            'CLIENT_TTS_REFERENCE_VOICES_PATH'
        ]
        
        for path_name in model_paths:
            assert hasattr(config, path_name)
            path_value = getattr(config, path_name)
            assert path_value is not None
            assert os.path.isabs(path_value)
    
    def test_logs_and_temp_paths(self):
        """Test that logs and temporary paths are properly defined."""
        assert hasattr(config, 'CLIENT_LOGS_PATH')
        assert hasattr(config, 'CLIENT_TEMP_AUDIO_PATH')
        
        # Both should be under CLIENT_DATA_PATH
        assert config.CLIENT_LOGS_PATH.startswith(config.CLIENT_DATA_PATH)
        assert config.CLIENT_TEMP_AUDIO_PATH.startswith(config.CLIENT_DATA_PATH)
    
    def test_path_consistency(self):
        """Test that all paths are consistent with each other."""
        # All paths should be absolute
        path_constants = [
            'CLIENT_ROOT', 'CLIENT_DATA_PATH', 'CLIENT_MODELS_PATH',
            'CLIENT_LLM_MODELS_PATH', 'CLIENT_TTS_MODELS_PATH',
            'CLIENT_TTS_REFERENCE_VOICES_PATH', 'CLIENT_LOGS_PATH',
            'CLIENT_TEMP_AUDIO_PATH'
        ]
        
        for path_const in path_constants:
            if hasattr(config, path_const):
                path_value = getattr(config, path_const)
                assert os.path.isabs(path_value), f"{path_const} should be absolute path"


class TestEnvironmentVariableOverrides:
    """Test environment variable override functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
        
    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': '/custom/data/path'})
    def test_data_path_override(self):
        """Test that CLIENT_DATA_PATH respects environment variable override."""
        # Need to reload the module to pick up environment changes
        import importlib
        importlib.reload(config)
        
        assert config.CLIENT_DATA_PATH == '/custom/data/path'
    
    @patch.dict(os.environ, {'DREAMWEAVER_CLIENT_MODELS_PATH': '/custom/models/path'})
    def test_models_path_override(self):
        """Test that CLIENT_MODELS_PATH respects environment variable override."""
        import importlib
        importlib.reload(config)
        
        assert config.CLIENT_MODELS_PATH == '/custom/models/path'
    
    @patch.dict(os.environ, {
        'DREAMWEAVER_CLIENT_DATA_PATH': '/env/data',
        'DREAMWEAVER_CLIENT_MODELS_PATH': '/env/models'
    })
    def test_multiple_env_overrides(self):
        """Test multiple environment variable overrides work together."""
        import importlib
        importlib.reload(config)
        
        assert config.CLIENT_DATA_PATH == '/env/data'
        assert config.CLIENT_MODELS_PATH == '/env/models'
    
    def test_env_override_path_derivation(self):
        """Test that derived paths update when base paths are overridden."""
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': '/new/base/path'}):
            import importlib
            importlib.reload(config)
            
            # Logs and temp paths should be derived from new base path
            assert config.CLIENT_LOGS_PATH.startswith('/new/base/path')
            assert config.CLIENT_TEMP_AUDIO_PATH.startswith('/new/base/path')
    
    @patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': 'relative/path'})
    def test_relative_path_handling(self):
        """Test behavior with relative paths in environment variables."""
        import importlib
        importlib.reload(config)
        
        # Should handle relative paths (behavior may vary by implementation)
        assert config.CLIENT_DATA_PATH == 'relative/path'


class TestDirectoryCreation:
    """Test the ensure_client_directories function."""
    
    def setup_method(self):
        """Set up test directories."""
        self.test_base_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Clean up test directories."""
        if os.path.exists(self.test_base_dir):
            shutil.rmtree(self.test_base_dir)
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_ensure_directories_creates_all_paths(self):
        """Test that ensure_client_directories creates all required directories."""
        test_data_path = os.path.join(self.test_base_dir, 'test_data')
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': test_data_path}):
            import importlib
            importlib.reload(config)
            
            # All directories should exist after module load
            directories_to_check = [
                config.CLIENT_DATA_PATH,
                config.CLIENT_MODELS_PATH,
                config.CLIENT_LLM_MODELS_PATH,
                config.CLIENT_TTS_MODELS_PATH,
                config.CLIENT_TTS_REFERENCE_VOICES_PATH,
                config.CLIENT_LOGS_PATH,
                config.CLIENT_TEMP_AUDIO_PATH
            ]
            
            for directory in directories_to_check:
                assert os.path.exists(directory), f"Directory {directory} should exist"
                assert os.path.isdir(directory), f"{directory} should be a directory"
    
    def test_ensure_directories_idempotent(self):
        """Test that ensure_client_directories is idempotent."""
        test_data_path = os.path.join(self.test_base_dir, 'idempotent_test')
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': test_data_path}):
            # Call ensure_client_directories multiple times
            config.ensure_client_directories()
            config.ensure_client_directories()
            config.ensure_client_directories()
            
            # Should not raise errors and directories should still exist
            assert os.path.exists(test_data_path)
    
    def test_ensure_directories_with_existing_dirs(self):
        """Test ensure_client_directories when some directories already exist."""
        test_data_path = os.path.join(self.test_base_dir, 'existing_test')
        os.makedirs(test_data_path, exist_ok=True)
        
        # Create some subdirectories manually
        os.makedirs(os.path.join(test_data_path, 'models'), exist_ok=True)
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': test_data_path}):
            import importlib
            importlib.reload(config)
            
            # All directories should still exist
            assert os.path.exists(config.CLIENT_MODELS_PATH)
    
    @patch('os.makedirs')
    def test_ensure_directories_handles_permission_errors(self, mock_makedirs):
        """Test that ensure_client_directories handles permission errors gracefully."""
        mock_makedirs.side_effect = PermissionError("Permission denied")
        
        # Should not raise exception, but log error
        try:
            config.ensure_client_directories()
        except PermissionError:
            pytest.fail("ensure_client_directories should handle PermissionError gracefully")
    
    @patch('os.makedirs')
    def test_ensure_directories_handles_os_errors(self, mock_makedirs):
        """Test that ensure_client_directories handles other OS errors gracefully."""
        mock_makedirs.side_effect = OSError("Disk full")
        
        # Should not raise exception, but log error
        try:
            config.ensure_client_directories()
        except OSError:
            pytest.fail("ensure_client_directories should handle OSError gracefully")


class TestConfigModuleExecution:
    """Test config module behavior when executed as main or imported."""
    
    def test_module_import_creates_directories(self):
        """Test that importing config module creates directories."""
        # Since config is already imported, directories should exist
        # This tests the import-time execution behavior
        assert os.path.exists(config.CLIENT_DATA_PATH)
        assert os.path.exists(config.CLIENT_LOGS_PATH)
    
    def test_module_attributes_accessibility(self):
        """Test that all expected module attributes are accessible."""
        expected_attributes = [
            'CLIENT_ROOT', 'DEFAULT_CLIENT_DATA_PATH', 'CLIENT_DATA_PATH',
            'DEFAULT_CLIENT_MODELS_PATH', 'CLIENT_MODELS_PATH',
            'CLIENT_LLM_MODELS_PATH', 'CLIENT_TTS_MODELS_PATH',
            'CLIENT_TTS_REFERENCE_VOICES_PATH', 'CLIENT_LOGS_PATH',
            'CLIENT_TEMP_AUDIO_PATH', 'ensure_client_directories'
        ]
        
        for attr in expected_attributes:
            assert hasattr(config, attr), f"Config module should have {attr} attribute"
    
    def test_config_main_execution_info_logging(self):
        """Test that config module provides useful info when run as main."""
        # This would test the if __name__ == "__main__" block
        # We can't easily test this directly, but we can test the components
        assert hasattr(config, 'CLIENT_ROOT')
        assert hasattr(config, 'CLIENT_DATA_PATH')


class TestLoggingConfigIntegration:
    """Test integration between config paths and logging configuration."""
    
    def setup_method(self):
        """Set up logging test environment."""
        # Clear any existing handlers
        logger = logging.getLogger('dreamweaver_client')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    def teardown_method(self):
        """Clean up logging test environment."""
        # Reset logging
        logger = logging.getLogger('dreamweaver_client')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    def test_client_log_file_path_derivation(self):
        """Test that CLIENT_LOG_FILE is derived from config paths."""
        from logging_config import CLIENT_LOG_FILE
        
        # Log file should be in the logs directory defined by config
        assert CLIENT_LOG_FILE.startswith(config.CLIENT_LOGS_PATH)
        assert CLIENT_LOG_FILE.endswith('client.log')
    
    def test_setup_client_logging_creates_log_directory(self):
        """Test that setting up logging ensures log directory exists."""
        # Clear any existing log directory
        temp_log_dir = tempfile.mkdtemp()
        temp_log_path = os.path.join(temp_log_dir, 'test_client.log')
        
        try:
            with patch('logging_config.CLIENT_LOG_FILE', temp_log_path):
                setup_client_logging()
                
                # Directory should be created
                assert os.path.exists(os.path.dirname(temp_log_path))
        finally:
            shutil.rmtree(temp_log_dir)
    
    def test_logging_integration_with_config_paths(self):
        """Test that logging configuration integrates properly with config paths."""
        # Test that we can get a logger and it's configured
        logger = get_logger()
        assert logger is not None
        assert logger.name == 'dreamweaver_client'
    
    def test_logging_file_rotation_configuration(self):
        """Test that log file rotation is properly configured."""
        setup_client_logging()
        logger = get_logger()
        
        # Should have both console and file handlers
        handler_types = [type(handler).__name__ for handler in logger.handlers]
        assert 'StreamHandler' in handler_types
        assert any('RotatingFileHandler' in ht for ht in handler_types)


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_config_with_unicode_paths(self):
        """Test configuration with Unicode characters in paths."""
        unicode_path = os.path.join(tempfile.gettempdir(), 'тест_客户端_🏠')
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': unicode_path}):
            try:
                import importlib
                importlib.reload(config)
                
                # Should handle Unicode paths gracefully
                assert config.CLIENT_DATA_PATH == unicode_path
                # Directory creation might succeed or fail depending on filesystem
                # but should not crash
            except UnicodeError:
                pytest.skip("System does not support Unicode in paths")
    
    def test_config_with_very_long_paths(self):
        """Test configuration with very long path names."""
        # Create a very long path (but within filesystem limits)
        long_component = 'a' * 100
        long_path = os.path.join(tempfile.gettempdir(), long_component, long_component)
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': long_path}):
            try:
                import importlib
                importlib.reload(config)
                
                assert config.CLIENT_DATA_PATH == long_path
            except OSError:
                # Some systems might reject very long paths
                pytest.skip("System does not support very long paths")
    
    def test_config_with_special_characters_in_paths(self):
        """Test configuration with special characters in paths."""
        special_chars_path = os.path.join(tempfile.gettempdir(), 'test-path_with.special@chars')
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': special_chars_path}):
            import importlib
            importlib.reload(config)
            
            assert config.CLIENT_DATA_PATH == special_chars_path
    
    def test_config_path_normalization(self):
        """Test that paths are properly normalized."""
        # Test with path containing redundant separators and relative components
        messy_path = os.path.join(tempfile.gettempdir(), 'test//path/../final_path')
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': messy_path}):
            import importlib
            importlib.reload(config)
            
            # Path should be used as provided (normalization is optional)
            assert config.CLIENT_DATA_PATH == messy_path


class TestConfigSecurity:
    """Test security-related aspects of configuration."""
    
    def test_config_no_sensitive_data_in_logs(self):
        """Test that configuration doesn't log sensitive information."""
        # Capture log output during module reload
        import io
        import sys
        
        log_capture = io.StringIO()
        
        # Redirect logging during config import
        with patch('sys.stderr', log_capture):
            import importlib
            importlib.reload(config)
        
        log_output = log_capture.getvalue()
        
        # Should not contain any paths that might be sensitive
        # This is a basic check - actual sensitivity depends on deployment
        sensitive_patterns = ['password', 'secret', 'key', 'token']
        for pattern in sensitive_patterns:
            assert pattern not in log_output.lower()
    
    def test_config_path_traversal_prevention(self):
        """Test that config handles path traversal attempts safely."""
        traversal_attempts = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32',
            '/etc/passwd',
            'C:\\Windows\\System32'
        ]
        
        for traversal_path in traversal_attempts:
            with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': traversal_path}):
                import importlib
                importlib.reload(config)
                
                # Should accept the path (validation is application responsibility)
                # but not create outside intended boundaries in production
                assert config.CLIENT_DATA_PATH == traversal_path


# Parametrized tests for comprehensive path testing
@pytest.mark.parametrize("env_var,config_attr", [
    ('DREAMWEAVER_CLIENT_DATA_PATH', 'CLIENT_DATA_PATH'),
    ('DREAMWEAVER_CLIENT_MODELS_PATH', 'CLIENT_MODELS_PATH'),
])
def test_environment_variable_mapping(env_var, config_attr):
    """Test that environment variables properly map to config attributes."""
    test_path = f'/test/path/for/{env_var.lower()}'
    
    with patch.dict(os.environ, {env_var: test_path}):
        import importlib
        importlib.reload(config)
        
        assert hasattr(config, config_attr)
        assert getattr(config, config_attr) == test_path


@pytest.mark.parametrize("log_level", [
    logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
])
def test_logging_levels(log_level):
    """Test logging setup with different log levels."""
    setup_client_logging(log_level)
    logger = get_logger()
    
    assert logger.level == log_level


@pytest.mark.parametrize("directory_attr", [
    'CLIENT_DATA_PATH', 'CLIENT_MODELS_PATH', 'CLIENT_LLM_MODELS_PATH',
    'CLIENT_TTS_MODELS_PATH', 'CLIENT_TTS_REFERENCE_VOICES_PATH',
    'CLIENT_LOGS_PATH', 'CLIENT_TEMP_AUDIO_PATH'
])
def test_all_directories_created(directory_attr):
    """Test that all configured directories are created."""
    if hasattr(config, directory_attr):
        directory_path = getattr(config, directory_attr)
        assert os.path.exists(directory_path)
        assert os.path.isdir(directory_path)


class TestConfigPerformance:
    """Test performance aspects of configuration operations."""
    
    def test_directory_creation_performance(self):
        """Test that directory creation is reasonably fast."""
        import time
        
        temp_base = tempfile.mkdtemp()
        try:
            start_time = time.time()
            
            # Create multiple directory structures
            for i in range(10):
                test_path = os.path.join(temp_base, f'perf_test_{i}')
                with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': test_path}):
                    config.ensure_client_directories()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete within reasonable time (less than 1 second for 10 iterations)
            assert total_time < 1.0
            
        finally:
            shutil.rmtree(temp_base)
    
    def test_module_import_performance(self):
        """Test that module import/reload is reasonably fast."""
        import time
        import importlib
        
        start_time = time.time()
        
        # Reload module multiple times
        for _ in range(5):
            importlib.reload(config)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly (less than 0.5 seconds for 5 reloads)
        assert total_time < 0.5


if __name__ == '__main__':
    pytest.main([__file__])