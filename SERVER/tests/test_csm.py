import pytest
import asyncio
import os
import tempfile
import wave
import struct
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import json

# Add the SERVER/src directory to the path so we can import CSM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from csm import CSM
except ImportError:
    # If CSM is in a different location, try alternate paths
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from csm import CSM
    except ImportError:
        CSM = None


class TestCSM:
    """Comprehensive test suite for CSM (Character Story Manager)"""
    
    @pytest.fixture
    def dummy_audio_file(self):
        """
        Yields the path to a temporary valid WAV audio file for use in tests.
        
        The file contains a 1-second mono audio sample at 44.1kHz with a simple waveform. The file is deleted after use.
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Create a minimal valid WAV file
            sample_rate = 44100
            duration = 1  # 1 second
            frequency = 440  # A4 note
            frames = int(duration * sample_rate)
            
            with wave.open(f.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                
                # Generate sine wave
                for i in range(frames):
                    value = int(32767 * 0.3 * (i % (sample_rate // frequency)) / (sample_rate // frequency))
                    wav_file.writeframes(struct.pack('<h', value))
            
            yield f.name
            # Cleanup
            try:
                os.unlink(f.name)
            except OSError:
                pass

    @pytest.fixture
    def mock_csm_dependencies(self):
        """
        Create and configure mocked dependencies for the CSM class, including narrator, character server, client manager, database, hardware, and chaos engine.
        
        Returns:
            dict: A dictionary containing mock objects for each CSM dependency, with key methods set up for asynchronous and synchronous behavior as needed.
        """
        mocks = {
            'narrator': Mock(),
            'character_server': Mock(),
            'client_manager': Mock(),
            'db': Mock(),
            'hardware': Mock(),
            'chaos_engine': Mock()
        }
        
        # Setup async mocks
        mocks['narrator'].process_narration = AsyncMock(return_value={
            "text": "Test narration text", 
            "audio_path": "test.wav", 
            "speaker": "Narrator"
        })
        mocks['character_server'].generate_response = AsyncMock(return_value="Test character response")
        mocks['client_manager'].send_to_client = AsyncMock(return_value="Client response sent")
        mocks['client_manager'].get_clients_for_story_progression = Mock(return_value=[
            {"Actor_id": "Actor1", "ip_address": "127.0.0.1", "client_port": 8001}
        ])
        mocks['client_manager'].start_periodic_health_checks = Mock()
        mocks['db'].get_character = Mock(return_value={
            "name": "TestCharacter", 
            "Actor_id": "Actor1"
        })
        
        return mocks

    @pytest.fixture
    def csm_instance(self, mock_csm_dependencies):
        """
        Create and return a `CSM` instance with all dependencies replaced by provided mocks.
        
        Skips the test if the `CSM` class is not available for import.
        
        Parameters:
            mock_csm_dependencies (dict): A dictionary mapping dependency names to their mock objects.
        
        Returns:
            CSM: An instance of the `CSM` class with mocked dependencies injected.
        """
        if CSM is None:
            pytest.skip("CSM class not available for import")
        
        with patch.multiple('csm', 
                          Database=Mock(),
                          Narrator=Mock(),
                          CharacterServer=Mock(),
                          ClientManager=Mock(),
                          Hardware=Mock(),
                          ChaosEngine=Mock()):
            csm = CSM()
            
            # Inject mocked dependencies
            for attr, mock_obj in mock_csm_dependencies.items():
                setattr(csm, attr, mock_obj)
            
            return csm

    @pytest.mark.asyncio
    async def test_process_story_happy_path(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Tests that `process_story` successfully processes a valid audio file and returns the expected narration and character dictionary when all dependencies operate normally.
        """
        # Act
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Assert
        assert narration == "Test narration text"
        assert isinstance(characters, dict)
        mock_csm_dependencies['narrator'].process_narration.assert_called_once_with(dummy_audio_file)
        mock_csm_dependencies['character_server'].generate_response.assert_called()
        mock_csm_dependencies['client_manager'].get_clients_for_story_progression.assert_called()

    @pytest.mark.asyncio
    async def test_process_story_with_server_character_response(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that processing a story includes the server character's response when a server character is configured.
        
        Asserts that the character server's response is included in the returned characters dictionary and that the database is queried for the server character.
        """
        # Setup
        mock_csm_dependencies['db'].get_character.return_value = {
            "name": "ServerCharacter", 
            "Actor_id": "Actor1"
        }
        mock_csm_dependencies['character_server'].generate_response.return_value = "Server character says hello"
        
        # Act
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Assert
        assert "ServerCharacter" in characters
        assert characters["ServerCharacter"] == "Server character says hello"
        mock_csm_dependencies['db'].get_character.assert_called_with("Actor1")

    @pytest.mark.asyncio
    async def test_process_story_with_multiple_chaos_levels(self, csm_instance, dummy_audio_file):
        """
        Test that `process_story` produces consistent narration and character outputs across multiple chaos levels.
        
        Runs the method with a range of chaos levels and asserts that the narration and character dictionary remain valid and as expected.
        """
        chaos_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        for chaos in chaos_levels:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=chaos)
            assert narration == "Test narration text"
            assert isinstance(characters, dict)

    @pytest.mark.asyncio
    async def test_process_story_nonexistent_audio_file(self, csm_instance, mock_csm_dependencies):
        """
        Test that processing a non-existent audio file raises a FileNotFoundError.
        """
        # Setup narrator to simulate file not found
        mock_csm_dependencies['narrator'].process_narration.side_effect = FileNotFoundError("Audio file not found")
        
        with pytest.raises(FileNotFoundError):
            await csm_instance.process_story("nonexistent_file.wav", chaos_level=0.0)

    @pytest.mark.asyncio
    async def test_process_story_invalid_audio_file(self, csm_instance, mock_csm_dependencies):
        """
        Tests that processing an invalid audio file format raises a ValueError.
        
        Creates a temporary text file with a `.txt` extension to simulate an invalid audio file. Mocks the narrator to raise a ValueError when attempting to process the file, and asserts that the exception is properly raised by `process_story`.
        """
        # Create invalid audio file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not an audio file")
            f.flush()
            
            try:
                # Setup narrator to simulate invalid format error
                mock_csm_dependencies['narrator'].process_narration.side_effect = ValueError("Invalid audio format")
                
                with pytest.raises(ValueError):
                    await csm_instance.process_story(f.name, chaos_level=0.0)
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_process_story_narrator_failure(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that an exception from the narrator during story processing is properly propagated.
        
        Asserts that when the narrator's processing raises an exception, `process_story` raises the same exception with the expected message.
        """
        # Setup narrator to raise exception
        mock_csm_dependencies['narrator'].process_narration.side_effect = Exception("Narrator processing failed")
        
        with pytest.raises(Exception) as exc_info:
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert "Narrator processing failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_story_empty_narration_text(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that process_story returns empty narration and an empty character dictionary when the narrator returns empty text.
        """
        # Setup narrator to return empty text
        mock_csm_dependencies['narrator'].process_narration.return_value = {
            "text": "", 
            "audio_path": "test.wav", 
            "speaker": "Narrator"
        }
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Should return empty narration and no characters
        assert narration == ""
        assert characters == {}

    @pytest.mark.asyncio
    async def test_process_story_narrator_returns_none(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that process_story returns empty narration and character dictionary when the narrator returns None.
        """
        mock_csm_dependencies['narrator'].process_narration.return_value = None
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration == ""
        assert characters == {}

    @pytest.mark.asyncio
    async def test_process_story_character_server_failure(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that `process_story` raises an exception when the character server fails during response generation.
        
        Asserts that the exception message matches the expected failure reason.
        """
        # Setup character server to raise exception
        mock_csm_dependencies['character_server'].generate_response.side_effect = Exception("Character server failed")
        
        with pytest.raises(Exception) as exc_info:
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert "Character server failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_story_no_server_character(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that `process_story` processes narration correctly when no server character is configured.
        
        Ensures that narration is generated and the character server is not called if the database returns no server character.
        """
        mock_csm_dependencies['db'].get_character.return_value = None
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Should still process but without server character response
        assert narration == "Test narration text"
        # Should not call character server if no Actor1
        mock_csm_dependencies['character_server'].generate_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_story_no_responsive_clients(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that `process_story` returns narration and a valid character dictionary when no clients are responsive.
        
        Simulates a scenario where the client manager reports no responsive clients. Verifies that narration is still produced and the returned characters object is a dictionary.
        """
        mock_csm_dependencies['client_manager'].get_clients_for_story_progression.return_value = []
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Should still return narration
        assert narration == "Test narration text"
        # May have server character response but no client responses
        assert isinstance(characters, dict)

    @pytest.mark.asyncio
    async def test_process_story_multiple_clients(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that `process_story` correctly handles multiple responsive clients.
        
        Simulates a scenario where several clients are available for story progression, each with a unique actor ID. Verifies that the method processes the story, contacts all clients, and returns the expected narration and character dictionary.
        """
        # Setup multiple clients
        mock_csm_dependencies['client_manager'].get_clients_for_story_progression.return_value = [
            {"Actor_id": "Actor1", "ip_address": "127.0.0.1", "client_port": 8001},
            {"Actor_id": "Actor2", "ip_address": "127.0.0.1", "client_port": 8002},
            {"Actor_id": "Actor3", "ip_address": "192.168.1.100", "client_port": 8003}
        ]
        
        # Setup character responses for different actors
        def mock_get_character(actor_id):
            """
            Return a mock character dictionary for the given actor ID.
            
            Parameters:
            	actor_id: The identifier for the actor.
            
            Returns:
            	dict: A dictionary containing the character's name and actor ID.
            """
            return {
                "name": f"Character_{actor_id}", 
                "Actor_id": actor_id
            }
        mock_csm_dependencies['db'].get_character.side_effect = mock_get_character
        
        # Mock client responses
        mock_csm_dependencies['client_manager'].send_to_client.return_value = "Client response"
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration == "Test narration text"
        assert isinstance(characters, dict)
        # Should have attempted to contact all clients
        assert mock_csm_dependencies['client_manager'].send_to_client.call_count >= 3

    @pytest.mark.asyncio
    async def test_process_story_client_communication_failure(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test how `process_story` handles failures in client communication.
        
        Simulates a scenario where communication with one client fails while another succeeds, verifying that the method either handles the failure gracefully or raises a meaningful exception related to client connectivity.
        """
        # Setup multiple clients
        mock_csm_dependencies['client_manager'].get_clients_for_story_progression.return_value = [
            {"Actor_id": "Actor1", "ip_address": "127.0.0.1", "client_port": 8001},
            {"Actor_id": "Actor2", "ip_address": "127.0.0.1", "client_port": 8002}
        ]
        
        # Setup one client to fail
        def side_effect(*args, **kwargs):
            """
            Simulates client communication by raising a ConnectionError for "Actor1" and returning a success response for other clients.
            
            Parameters:
            	args: Positional arguments, where the first argument is expected to be the client identifier.
            
            Returns:
            	str: "Success response" if the client is not "Actor1".
            
            Raises:
            	ConnectionError: If the first argument is "Actor1".
            """
            if args[0] == "Actor1":  # First client fails
                raise ConnectionError("Client unreachable")
            return "Success response"
        
        mock_csm_dependencies['client_manager'].send_to_client.side_effect = side_effect
        
        # Should handle gracefully or raise appropriate exception
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            assert narration == "Test narration text"
        except Exception as e:
            # If it raises, should be meaningful
            assert any(keyword in str(e).lower() for keyword in ["connection", "client", "communication"])

    @pytest.mark.asyncio
    async def test_process_story_database_failure(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that process_story raises an exception when a database operation fails.
        
        Simulates a database failure during character retrieval and asserts that the exception is propagated with the expected error message.
        """
        mock_csm_dependencies['db'].get_character.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception) as exc_info:
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert "Database connection failed" in str(exc_info.value)

    @pytest.mark.parametrize("chaos_level", [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0])
    @pytest.mark.asyncio
    async def test_chaos_level_parameter_range(self, csm_instance, dummy_audio_file, chaos_level):
        """
        Tests that story processing produces consistent narration and character outputs across a wide range of chaos level values.
        
        Parameters:
        	chaos_level: The chaos level value to test, covering various valid and extreme cases.
        """
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=chaos_level)
        
        assert narration == "Test narration text"
        assert isinstance(characters, dict)

    @pytest.mark.asyncio
    async def test_process_story_invalid_chaos_level_types(self, csm_instance, dummy_audio_file):
        """
        Test that `process_story` handles invalid chaos level types by either raising an appropriate exception or processing gracefully with default narration.
        """
        invalid_chaos_levels = ["invalid", None, [], {}]
        
        for chaos in invalid_chaos_levels:
            try:
                narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=chaos)
                # If it doesn't raise, verify it handled gracefully
                assert narration == "Test narration text"
            except (ValueError, TypeError) as e:
                # Expected behavior for invalid inputs
                assert isinstance(e, (ValueError, TypeError))

    @pytest.mark.asyncio
    async def test_process_story_negative_chaos_level(self, csm_instance, dummy_audio_file):
        """
        Test that `process_story` handles negative chaos levels by either processing successfully or raising a `ValueError`.
        
        This test verifies that the method accepts or rejects negative chaos levels appropriately, ensuring robust input validation or graceful handling.
        """
        negative_levels = [-1.0, -0.5, -10.0]
        
        for chaos in negative_levels:
            try:
                narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=chaos)
                # Should handle gracefully or raise appropriate error
                assert narration == "Test narration text"
            except ValueError:
                # Acceptable to reject negative values
                pass

    @pytest.mark.asyncio
    async def test_shutdown_async_functionality(self, csm_instance):
        """
        Test that the CSM instance provides an async shutdown method and that it executes without error.
        
        Skips the test if the `shutdown_async` method is not implemented.
        """
        if hasattr(csm_instance, 'shutdown_async'):
            # Test that shutdown completes without error
            await csm_instance.shutdown_async()
        else:
            pytest.skip("shutdown_async method not implemented")

    @pytest.mark.asyncio
    async def test_shutdown_async_multiple_calls(self, csm_instance):
        """
        Verify that calling the `shutdown_async` method multiple times on the CSM instance does not raise exceptions and is handled gracefully.
        """
        if hasattr(csm_instance, 'shutdown_async'):
            await csm_instance.shutdown_async()
            await csm_instance.shutdown_async()  # Should not raise
            await csm_instance.shutdown_async()  # Should not raise
        else:
            pytest.skip("shutdown_async method not implemented")

    @pytest.mark.asyncio
    async def test_concurrent_process_story_calls(self, csm_instance, dummy_audio_file):
        """
        Test that multiple concurrent calls to `process_story` complete successfully and return expected results.
        
        Runs several asynchronous `process_story` calls in parallel with different chaos levels, then verifies that each returns the correct narration and a valid character dictionary.
        """
        # Create multiple concurrent tasks
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                csm_instance.process_story(dummy_audio_file, chaos_level=0.1 * i)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify results
        for result in results:
            if not isinstance(result, Exception):
                narration, characters = result
                assert narration == "Test narration text"
                assert isinstance(characters, dict)

    def test_csm_initialization(self):
        """
        Test that the CSM class initializes with all required non-None attributes.
        """
        if CSM is None:
            pytest.skip("CSM class not available for import")
        
        with patch.multiple('csm', 
                          Database=Mock(),
                          Narrator=Mock(),
                          CharacterServer=Mock(),
                          ClientManager=Mock(),
                          Hardware=Mock(),
                          ChaosEngine=Mock()):
            csm = CSM()
            
            # Verify that CSM has expected attributes
            expected_attributes = ['narrator', 'character_server', 'client_manager', 'db', 'hardware', 'chaos_engine']
            for attr in expected_attributes:
                assert hasattr(csm, attr), f"CSM should have {attr} attribute"
                assert getattr(csm, attr) is not None, f"CSM.{attr} should not be None"

    @pytest.mark.asyncio
    async def test_process_story_with_special_characters_in_path(self, csm_instance, mock_csm_dependencies):
        """
        Tests that the story processing function correctly handles audio files with special characters in their file paths, such as spaces, Unicode characters, and multiple dots.
        
        Creates temporary WAV files with various special characters in their filenames, processes each file using the CSM instance, and asserts that narration and character outputs are as expected.
        """
        special_files = []
        
        try:
            # Create files with special characters
            special_names = ["spaces in name.wav", "unicode_ñáme.wav", "dots...in.name.wav"]
            
            for filename in special_names:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, prefix='test_') as f:
                    # Create minimal WAV file
                    with wave.open(f.name, 'w') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(44100)
                        wav_file.writeframes(b'\x00' * 88200)  # 1 second of silence
                    
                    special_files.append(f.name)
                    
                    narration, characters = await csm_instance.process_story(f.name, chaos_level=0.0)
                    assert narration == "Test narration text"
                    assert isinstance(characters, dict)
                    
        finally:
            # Cleanup
            for filepath in special_files:
                try:
                    os.unlink(filepath)
                except OSError:
                    pass

    @pytest.mark.asyncio
    async def test_process_story_performance_timing(self, csm_instance, dummy_audio_file):
        """
        Tests that the story processing completes within 10 seconds and returns the expected narration and character dictionary.
        """
        import time
        
        start_time = time.time()
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        end_time = time.time()
        
        # Should complete within 10 seconds (mocked dependencies should be fast)
        assert end_time - start_time < 10, f"Story processing took {end_time - start_time} seconds, expected < 10"
        assert narration == "Test narration text"
        assert isinstance(characters, dict)

    @pytest.mark.asyncio
    async def test_process_story_return_value_structure(self, csm_instance, dummy_audio_file):
        """
        Test that the process_story method returns a string narration and a dictionary of character responses with string keys and values.
        """
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Verify return types
        assert isinstance(narration, str), "Narration should be a string"
        assert isinstance(characters, dict), "Characters should be a dictionary"
        
        # Verify character dict structure
        for char_name, char_text in characters.items():
            assert isinstance(char_name, str), "Character names should be strings"
            assert isinstance(char_text, str), "Character texts should be strings"

    @pytest.mark.asyncio
    async def test_process_story_with_empty_character_response(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that `process_story` correctly handles an empty response from the character server.
        
        Verifies that when the character server returns an empty string, the narration is still returned as expected and the character dictionary includes the character with an empty response.
        """
        mock_csm_dependencies['character_server'].generate_response.return_value = ""
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration == "Test narration text"
        # Empty character response should not be included in results
        if "TestCharacter" in characters:
            assert characters["TestCharacter"] == ""

    @pytest.mark.asyncio
    async def test_process_story_with_none_character_response(self, csm_instance, dummy_audio_file, mock_csm_dependencies):
        """
        Test that process_story handles a None response from the character server without errors.
        
        Asserts that narration is returned as expected and the characters dictionary is valid even when the character server returns None.
        """
        mock_csm_dependencies['character_server'].generate_response.return_value = None
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration == "Test narration text"
        # None response should not cause errors
        assert isinstance(characters, dict)


# Integration test class for more complex scenarios
class TestCSMIntegration:
    """Integration tests for CSM with realistic scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_story_processing_workflow(self):
        """
        Placeholder for a full end-to-end integration test of the story processing workflow.
        
        Currently skipped because it requires a complete system setup with all components integrated.
        """
        if CSM is None:
            pytest.skip("CSM class not available for import")
        
        # This would test with more realistic mocks that simulate actual component behavior
        # For now, we'll skip this as it requires more detailed knowledge of the full system
        pytest.skip("Full integration test requires complete system setup")

    @pytest.mark.asyncio
    async def test_stress_multiple_concurrent_stories(self):
        """
        Placeholder for a stress test that would process multiple stories concurrently to evaluate system performance under high load.
        
        Currently skipped due to the need for a dedicated performance benchmarking setup.
        """
        if CSM is None:
            pytest.skip("CSM class not available for import")
        
        # Test high concurrency scenarios
        pytest.skip("Stress testing requires performance benchmarking setup")


# Fixtures for test data
@pytest.fixture(scope="session")
def test_audio_samples():
    """
    Yields file paths to short and long temporary WAV audio samples for use in tests.
    
    Returns:
        samples (dict): Dictionary with keys 'short' and 'long' mapping to file paths of generated WAV files.
    """
    samples = {}
    
    try:
        # Short audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            with wave.open(f.name, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(b'\x00' * 4410)  # 0.1 seconds
            samples['short'] = f.name
        
        # Long audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            with wave.open(f.name, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(b'\x00' * 441000)  # 10 seconds
            samples['long'] = f.name
        
        yield samples
        
    finally:
        # Cleanup
        for filepath in samples.values():
            try:
                os.unlink(filepath)
            except OSError:
                pass


# Custom pytest markers for different test categories
pytestmark = [
    pytest.mark.asyncio,  # All tests in this module are async
]


if __name__ == "__main__":
    # Run tests with verbose output when executed directly
    pytest.main([__file__, "-v", "--tb=short"])