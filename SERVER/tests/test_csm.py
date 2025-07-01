import unittest
import asyncio
import os
import tempfile
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add the SERVER src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from csm import CSM
except ImportError:
    print("Warning: Could not import CSM. Ensure the CSM module is available.")
    CSM = None


class TestCSMInitialization(unittest.TestCase):
    """Test CSM initialization and basic setup."""
    
    def setUp(self):
        """
        Prepare the test environment by creating a new CSM instance, or skip the test if the CSM module is unavailable.
        """
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def test_csm_initialization(self):
        """
        Verify that initializing the CSM instance creates all required component attributes.
        """
        self.assertIsNotNone(self.csm)
        self.assertTrue(hasattr(self.csm, 'narrator'))
        self.assertTrue(hasattr(self.csm, 'character_server'))
        self.assertTrue(hasattr(self.csm, 'client_manager'))
        self.assertTrue(hasattr(self.csm, 'db'))
    
    def test_csm_multiple_instances(self):
        """
        Verify that creating multiple CSM instances results in distinct, independent objects.
        """
        csm1 = CSM()
        csm2 = CSM()
        self.assertIsNot(csm1, csm2)


class TestCSMProcessStory(unittest.TestCase):
    """Test the main process_story functionality."""
    
    def setUp(self):
        """
        Initializes the test environment by creating a CSM instance and setting up dummy audio data for use in tests.
        
        Skips the test if the CSM module is unavailable.
        """
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
        self.test_audio_file = "test_audio.wav"
        self.dummy_narration = {
            "text": "Test narration text",
            "audio_path": self.test_audio_file,
            "speaker": "Narrator"
        }
    
    def tearDown(self):
        """
        Cleans up the CSM instance after each test by shutting it down asynchronously if it exists.
        """
        if hasattr(self, 'csm'):
            # Attempt to clean up CSM resources
            try:
                asyncio.run(self.csm.shutdown_async())
            except:
                pass
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server') 
    @patch('csm.CSM.client_manager')
    @patch('csm.CSM.db')
    def test_process_story_happy_path(self, mock_db, mock_cm, mock_cs, mock_narrator):
        """
        Test that `process_story` completes successfully when all dependencies operate normally.
        
        Mocks all major components to simulate a successful story processing pipeline and verifies that the returned narration and character data are valid and that the appropriate methods are called.
        """
        # Setup mocks
        mock_narrator.process_narration = AsyncMock(return_value=self.dummy_narration)
        mock_cs.generate_response = AsyncMock(return_value="Character response")
        mock_cm.get_clients_for_story_progression = Mock(return_value=[
            {"Actor_id": "Actor1", "ip_address": "127.0.0.1", "client_port": 8001}
        ])
        mock_cm.send_to_client = AsyncMock(return_value="Client response")
        mock_db.get_character = Mock(return_value={
            "name": "TestCharacter", 
            "Actor_id": "Actor1"
        })
        
        async def run_test():
            """
            Runs an asynchronous test of the CSM's process_story method with a test audio file and chaos level, verifying output and that key component mocks are called as expected.
            """
            narration, characters = await self.csm.process_story(
                self.test_audio_file, 
                chaos_level=0.5
            )
            
            # Verify results
            self.assertIsNotNone(narration)
            self.assertIsInstance(characters, dict)
            
            # Verify mocks were called
            mock_narrator.process_narration.assert_called_once_with(self.test_audio_file)
            mock_cs.generate_response.assert_called()
            mock_cm.get_clients_for_story_progression.assert_called_once()
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    def test_process_story_narrator_failure(self, mock_narrator):
        """
        Test that process_story raises an exception when the narrator component fails.
        
        This test mocks the narrator's process_narration method to raise an exception and verifies that process_story propagates the error.
        """
        mock_narrator.process_narration = AsyncMock(side_effect=Exception("Narrator error"))
        
        async def run_test():
            """
            Asynchronously runs a test that expects `process_story` to raise an exception, and asserts that the exception message contains 'Narrator error'.
            """
            with self.assertRaises(Exception) as context:
                await self.csm.process_story(self.test_audio_file)
            self.assertIn("Narrator error", str(context.exception))
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    def test_process_story_character_server_failure(self, mock_cs, mock_narrator):
        """
        Test that process_story raises an exception when the character server fails.
        
        Mocks the character server to raise an exception during response generation and verifies that process_story propagates the error with the expected message.
        """
        mock_narrator.process_narration = AsyncMock(return_value=self.dummy_narration)
        mock_cs.generate_response = AsyncMock(side_effect=Exception("Character server error"))
        
        async def run_test():
            """
            Asynchronously runs a test that expects `process_story` to raise an exception containing 'Character server error'.
            """
            with self.assertRaises(Exception) as context:
                await self.csm.process_story(self.test_audio_file)
            self.assertIn("Character server error", str(context.exception))
        
        asyncio.run(run_test())
    
    def test_process_story_invalid_audio_file(self):
        """
        Test that process_story raises an exception when given a non-existent audio file.
        """
        async def run_test():
            """
            Asynchronously tests that processing a nonexistent audio file raises a FileNotFoundError or Exception.
            """
            with self.assertRaises((FileNotFoundError, Exception)):
                await self.csm.process_story("nonexistent_file.wav")
        
        asyncio.run(run_test())
    
    def test_process_story_invalid_input_types(self):
        """
        Test that process_story raises an exception when called with invalid input types.
        
        Verifies that passing None, an integer, a list, a dictionary, or an empty string to process_story results in a TypeError, ValueError, or other exception.
        """
        invalid_inputs = [None, 123, [], {}, ""]
        
        async def run_test():
            """
            Asynchronously tests that `process_story` raises an exception when called with invalid input types.
            """
            for invalid_input in invalid_inputs:
                with self.assertRaises((TypeError, ValueError, Exception)):
                    await self.csm.process_story(invalid_input)
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    def test_process_story_chaos_level_variations(self, mock_cm, mock_cs, mock_narrator):
        """
        Tests that the process_story method returns valid narration and character data across a range of chaos levels from 0.0 to 1.0.
        """
        mock_narrator.process_narration = AsyncMock(return_value=self.dummy_narration)
        mock_cs.generate_response = AsyncMock(return_value="Character response")
        mock_cm.get_clients_for_story_progression = Mock(return_value=[])
        
        chaos_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        async def run_test():
            """
            Runs the `process_story` method for each chaos level and asserts valid narration and character outputs.
            
            Iterates through predefined chaos levels, invoking `process_story` asynchronously with a test audio file, and verifies that the narration is not None and the characters result is a dictionary.
            """
            for chaos_level in chaos_levels:
                narration, characters = await self.csm.process_story(
                    self.test_audio_file, 
                    chaos_level=chaos_level
                )
                self.assertIsNotNone(narration)
                self.assertIsInstance(characters, dict)
        
        asyncio.run(run_test())
    
    def test_process_story_invalid_chaos_levels(self):
        """
        Tests that `process_story` raises a `ValueError` or `TypeError` when called with invalid chaos level values.
        """
        invalid_chaos_levels = [-0.1, 1.1, 2.0, "invalid", None, float('inf')]
        
        async def run_test():
            """
            Runs the process_story method with invalid chaos levels and asserts that a ValueError or TypeError is raised for each case.
            """
            for chaos_level in invalid_chaos_levels:
                with self.assertRaises((ValueError, TypeError)):
                    await self.csm.process_story(self.test_audio_file, chaos_level=chaos_level)
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    @patch('csm.CSM.db')
    def test_process_story_no_clients(self, mock_db, mock_cm, mock_cs, mock_narrator):
        """
        Test that process_story returns valid narration and character data when no clients are available.
        
        Mocks the client manager to return an empty client list and verifies that process_story completes successfully, returning a narration and a dictionary for characters.
        """
        mock_narrator.process_narration = AsyncMock(return_value=self.dummy_narration)
        mock_cs.generate_response = AsyncMock(return_value="Character response")
        mock_cm.get_clients_for_story_progression = Mock(return_value=[])
        
        async def run_test():
            """
            Runs the `process_story` method asynchronously and asserts that narration is returned and characters is a dictionary.
            """
            narration, characters = await self.csm.process_story(self.test_audio_file)
            self.assertIsNotNone(narration)
            self.assertIsInstance(characters, dict)
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    @patch('csm.CSM.db')
    def test_process_story_multiple_clients(self, mock_db, mock_cm, mock_cs, mock_narrator):
        """
        Test that process_story correctly handles scenarios with multiple clients.
        
        This test mocks the narrator, character server, client manager, and database to simulate three distinct clients. It verifies that process_story returns narration and a character dictionary containing entries for each client.
        """
        mock_narrator.process_narration = AsyncMock(return_value=self.dummy_narration)
        mock_cs.generate_response = AsyncMock(return_value="Character response")
        mock_cm.get_clients_for_story_progression = Mock(return_value=[
            {"Actor_id": "Actor1", "ip_address": "127.0.0.1", "client_port": 8001},
            {"Actor_id": "Actor2", "ip_address": "127.0.0.1", "client_port": 8002},
            {"Actor_id": "Actor3", "ip_address": "127.0.0.1", "client_port": 8003}
        ])
        mock_cm.send_to_client = AsyncMock(return_value="Client response")
        mock_db.get_character = Mock(side_effect=lambda actor_id: {
            "name": f"Character_{actor_id}", 
            "Actor_id": actor_id
        })
        
        async def run_test():
            """
            Runs an asynchronous test of the CSM's story processing, asserting that narration is returned and exactly three characters are present.
            """
            narration, characters = await self.csm.process_story(self.test_audio_file)
            self.assertIsNotNone(narration)
            self.assertIsInstance(characters, dict)
            self.assertEqual(len(characters), 3)
        
        asyncio.run(run_test())


class TestCSMErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """
        Prepare the test environment by creating a new CSM instance, or skip the test if the CSM module is unavailable.
        """
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def tearDown(self):
        """
        Cleans up the test fixture by asynchronously shutting down the CSM instance after each test.
        """
        try:
            asyncio.run(self.csm.shutdown_async())
        except:
            pass
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    @patch('csm.CSM.db')
    def test_database_connection_failure(self, mock_db, mock_cm, mock_cs, mock_narrator):
        """
        Test that `process_story` raises an exception when a database connection failure occurs during character retrieval.
        
        Mocks the database to simulate a connection failure and verifies that the exception is propagated with the expected error message.
        """
        mock_narrator.process_narration = AsyncMock(return_value={
            "text": "Test", "audio_path": "test.wav", "speaker": "Narrator"
        })
        mock_cs.generate_response = AsyncMock(return_value="Response")
        mock_cm.get_clients_for_story_progression = Mock(return_value=[
            {"Actor_id": "Actor1", "ip_address": "127.0.0.1", "client_port": 8001}
        ])
        mock_cm.send_to_client = AsyncMock(return_value="Sent")
        mock_db.get_character = Mock(side_effect=Exception("Database connection failed"))
        
        async def run_test():
            """
            Asynchronously runs a test to verify that processing a story raises an exception with a database connection failure message.
            
            Asserts that calling `process_story` with "test.wav" raises an exception containing "Database connection failed".
            """
            with self.assertRaises(Exception) as context:
                await self.csm.process_story("test.wav")
            self.assertIn("Database connection failed", str(context.exception))
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    @patch('csm.CSM.db')
    def test_client_communication_failure(self, mock_db, mock_cm, mock_cs, mock_narrator):
        """
        Test that `process_story` handles exceptions raised during client communication, such as when sending data to a client fails. Verifies that the method either returns partial results or raises an appropriate exception.
        """
        mock_narrator.process_narration = AsyncMock(return_value={
            "text": "Test", "audio_path": "test.wav", "speaker": "Narrator"
        })
        mock_cs.generate_response = AsyncMock(return_value="Response")
        mock_cm.get_clients_for_story_progression = Mock(return_value=[
            {"Actor_id": "Actor1", "ip_address": "127.0.0.1", "client_port": 8001}
        ])
        mock_cm.send_to_client = AsyncMock(side_effect=Exception("Client unreachable"))
        mock_db.get_character = Mock(return_value={"name": "Test", "Actor_id": "Actor1"})
        
        async def run_test():
            # Should handle client communication errors gracefully
            """
            Runs a test to verify that client communication errors during story processing are handled gracefully.
            
            The test calls the asynchronous `process_story` method and asserts that either partial results are returned or an appropriate exception message is raised.
            """
            try:
                narration, characters = await self.csm.process_story("test.wav")
                # Might return partial results or raise exception
                self.assertIsNotNone(narration)
            except Exception as e:
                self.assertIn("Client unreachable", str(e))
        
        asyncio.run(run_test())
    
    def test_concurrent_process_story_calls(self):
        """
        Tests that multiple concurrent calls to `process_story` are handled without deadlocks, even when all calls result in exceptions due to missing audio files.
        """
        async def run_test():
            """
            Runs three concurrent `process_story` calls with different audio files and asserts that all raise exceptions, verifying that concurrent execution does not cause deadlocks.
            
            Returns:
                None
            """
            tasks = []
            for i in range(3):
                task = asyncio.create_task(
                    self.csm.process_story(f"test_audio_{i}.wav")
                )
                tasks.append(task)
            
            # Should handle concurrent calls without deadlocks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.assertEqual(len(results), 3)
            
            # All should be exceptions (file not found) but no deadlocks
            for result in results:
                self.assertIsInstance(result, Exception)
        
        asyncio.run(run_test())


class TestCSMShutdown(unittest.TestCase):
    """Test CSM shutdown functionality."""
    
    def setUp(self):
        """
        Prepare the test environment by creating a new CSM instance, or skip the test if the CSM module is unavailable.
        """
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def test_shutdown_async(self):
        """
        Tests that the CSM's asynchronous shutdown method properly calls the shutdown methods of its components.
        """
        async def run_test():
            # Mock shutdown methods if they exist
            """
            Asynchronously tests that the shutdown methods of the CSM's components are called exactly once during shutdown.
            """
            with patch.object(self.csm, 'narrator', Mock()) as mock_narrator, \
                 patch.object(self.csm, 'character_server', Mock()) as mock_cs, \
                 patch.object(self.csm, 'client_manager', Mock()) as mock_cm:
                
                mock_narrator.shutdown = AsyncMock()
                mock_cs.shutdown = AsyncMock()
                mock_cm.shutdown = AsyncMock()
                
                await self.csm.shutdown_async()
                
                # Verify shutdown methods called if they exist
                if hasattr(mock_narrator, 'shutdown'):
                    mock_narrator.shutdown.assert_called_once()
                if hasattr(mock_cs, 'shutdown'):
                    mock_cs.shutdown.assert_called_once()
                if hasattr(mock_cm, 'shutdown'):
                    mock_cm.shutdown.assert_called_once()
        
        asyncio.run(run_test())
    
    def test_shutdown_twice(self):
        """
        Verify that calling the shutdown method multiple times on the CSM instance does not raise exceptions.
        """
        async def run_test():
            """
            Asynchronously calls the CSM shutdown method twice to verify that repeated shutdowns do not raise exceptions.
            """
            await self.csm.shutdown_async()
            # Should not raise exception on second call
            await self.csm.shutdown_async()
        
        asyncio.run(run_test())


class TestCSMEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """
        Prepare the test environment by creating a new CSM instance, or skip the test if the CSM module is unavailable.
        """
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def tearDown(self):
        """
        Cleans up the test fixture by asynchronously shutting down the CSM instance after each test.
        """
        try:
            asyncio.run(self.csm.shutdown_async())
        except:
            pass
    
    def test_very_long_audio_filename(self):
        """
        Tests that processing a story with an excessively long audio filename raises an OSError or other exception.
        """
        long_filename = "a" * 1000 + ".wav"
        
        async def run_test():
            """
            Runs the process_story method with a very long filename and asserts that an OSError or generic Exception is raised.
            """
            with self.assertRaises((OSError, Exception)):
                await self.csm.process_story(long_filename)
        
        asyncio.run(run_test())
    
    def test_special_characters_in_filename(self):
        """
        Tests that `process_story` handles audio filenames containing spaces, dashes, underscores, and dots, expecting exceptions for non-existent files.
        """
        special_filenames = [
            "file with spaces.wav",
            "file-with-dashes.wav",
            "file_with_underscores.wav",
            "file.with.dots.wav"
        ]
        
        async def run_test():
            """
            Runs the `process_story` method asynchronously for each filename in `special_filenames`, ignoring exceptions for non-existent files.
            """
            for filename in special_filenames:
                try:
                    await self.csm.process_story(filename)
                except (FileNotFoundError, Exception):
                    pass  # Expected for non-existent files
        
        asyncio.run(run_test())
    
    def test_unicode_filenames(self):
        """
        Tests that the `process_story` method handles filenames containing Unicode characters, expecting exceptions for non-existent files.
        """
        unicode_filenames = [
            "测试.wav",
            "tëst.wav",
            "тест.wav"
        ]
        
        async def run_test():
            """
            Runs the `process_story` method asynchronously for each filename in `unicode_filenames`, ignoring exceptions for non-existent files.
            """
            for filename in unicode_filenames:
                try:
                    await self.csm.process_story(filename)
                except (FileNotFoundError, Exception):
                    pass  # Expected for non-existent files
        
        asyncio.run(run_test())


class TestCSMPerformance(unittest.TestCase):
    """Performance and timeout tests."""
    
    def setUp(self):
        """
        Prepare the test environment by creating a new CSM instance, or skip the test if the CSM module is unavailable.
        """
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def tearDown(self):
        """
        Cleans up the test fixture by asynchronously shutting down the CSM instance after each test.
        """
        try:
            asyncio.run(self.csm.shutdown_async())
        except:
            pass
    
    @patch('csm.CSM.narrator')
    def test_timeout_handling(self, mock_narrator):
        """
        Tests that the `process_story` method raises an `asyncio.TimeoutError` when narration processing exceeds the specified timeout.
        """
        async def slow_process(audio_file):
            """
            Simulates a slow narration processing operation by delaying for 10 seconds.
            
            Parameters:
            	audio_file (str): Path to the audio file to process.
            
            Returns:
            	dict: A dictionary containing the narration text, audio file path, and speaker name.
            """
            await asyncio.sleep(10)  # Simulate very slow processing
            return {"text": "Slow result", "audio_path": audio_file, "speaker": "Narrator"}
        
        mock_narrator.process_narration = slow_process
        
        async def run_test():
            """
            Runs the `process_story` method with a 1-second timeout and asserts that an `asyncio.TimeoutError` is raised.
            """
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self.csm.process_story("test.wav"),
                    timeout=1.0
                )
        
        asyncio.run(run_test())


# Original integration test (preserved and enhanced)
class TestCSMIntegration(unittest.TestCase):
    """Integration test for CSM process_story - preserved from original test."""
    
    def setUp(self):
        """
        Prepare the integration test environment by creating a CSM instance and setting a dummy audio filename.
        
        Skips the test if the CSM module is unavailable.
        """
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
        self.dummy_audio_file = "csm_test_narration.wav"
    
    def tearDown(self):
        """
        Cleans up resources after integration tests by shutting down the CSM instance and removing the dummy audio file if it exists.
        """
        try:
            asyncio.run(self.csm.shutdown_async())
        except:
            pass
        
        # Clean up test file
        if os.path.exists(self.dummy_audio_file):
            try:
                os.remove(self.dummy_audio_file)
            except:
                pass
        
    def test_csm_process_story_integration(self):
        """
        Performs an integration test of the CSM's asynchronous `process_story` method using a dummy audio file and mocked component methods.
        
        This test creates a minimal WAV file if needed, mocks all major CSM dependencies to return controlled test data, invokes `process_story`, prints and validates the results, restores original methods, and shuts down the CSM instance.
        """
        async def run_integration_test():
            """
            Runs an asynchronous integration test for the CSM's process_story method using mocked dependencies.
            
            This test creates a dummy audio file if needed, mocks all major CSM components to return predictable results, invokes process_story, prints the outputs, validates the results, restores the original methods, and shuts down the CSM instance.
            """
            print("Testing CSM process_story (async)...")
            
            # Create a dummy audio file if it doesn't exist
            if not os.path.exists(self.dummy_audio_file):
                # Create minimal WAV file for testing
                wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
                with open(self.dummy_audio_file, 'wb') as f:
                    f.write(wav_header)
            
            # Preserve original mocking approach from existing test
            original_narrator_process = self.csm.narrator.process_narration
            async def mock_narrator_process(audio_filepath):
                """
                Simulates narrator processing by returning a mock narration result for the given audio file path.
                
                Parameters:
                    audio_filepath (str): Path to the input audio file.
                
                Returns:
                    dict: A dictionary containing mock narration text, the provided audio file path, and the speaker name.
                """
                return {"text": "This is a test narration from mock.", "audio_path": audio_filepath, "speaker": "Narrator"}
            self.csm.narrator.process_narration = mock_narrator_process
            
            original_cs_gen_response = self.csm.character_server.generate_response
            async def mock_cs_gen_response(narration, other_texts):
                """
                Asynchronously returns a mock character response for testing purposes.
                
                Parameters:
                    narration: The narration text input.
                    other_texts: Additional text inputs for the response.
                
                Returns:
                    str: A fixed mock response string.
                """
                return "Actor1 says hello asynchronously!"
            self.csm.character_server.generate_response = mock_cs_gen_response
            
            original_cm_send_to_client = self.csm.client_manager.send_to_client
            async def mock_cm_send_to_client(client_actor_id, client_ip, client_port, narration, character_texts):
                """
                Asynchronously returns a mock response simulating a client sending a message.
                
                Parameters:
                	client_actor_id: Identifier for the client actor.
                	client_ip: IP address of the client.
                	client_port: Port number of the client.
                	narration: Narration text to be sent.
                	character_texts: Character texts to be sent.
                
                Returns:
                	A string indicating the client actor sent a message via the async mock.
                """
                return f"{client_actor_id} says hi via async mock!"
            self.csm.client_manager.send_to_client = mock_cm_send_to_client
            
            original_cm_get_clients = self.csm.client_manager.get_clients_for_story_progression
            def mock_cm_get_clients():
                """
                Return a list containing a single mock client dictionary for testing purposes.
                
                Returns:
                    list: A list with one dictionary representing a mock client, including actor ID, IP address, and port.
                """
                return [{"Actor_id": "Actor_TestClient", "ip_address": "127.0.0.1", "client_port": 8001}]
            self.csm.client_manager.get_clients_for_story_progression = mock_cm_get_clients
            
            original_db_get_char = self.csm.db.get_character
            def mock_db_get_char(Actor_id):
                """
                Return a mock character dictionary for a given actor ID.
                
                Parameters:
                    Actor_id (str): The identifier of the actor.
                
                Returns:
                    dict or None: A dictionary with character information if the actor ID matches a known test value; otherwise, None.
                """
                if Actor_id == "Actor1":
                    return {"name": "ServerTestChar", "Actor_id": "Actor1"}
                if Actor_id == "Actor_TestClient":
                    return {"name": "RemoteTestChar", "Actor_id": "Actor_TestClient"}
                return None
            self.csm.db.get_character = mock_db_get_char
            
            print("Processing story with CSM...")
            narration, characters = await self.csm.process_story(self.dummy_audio_file, chaos_level=0.0)
            
            print("\n--- CSM Test Results ---")
            print(f"Narrator: {narration}")
            print("Characters:")
            for char, text in characters.items():
                print(f"  {char}: {text}")
            
            # Validate results
            self.assertIsNotNone(narration)
            self.assertIsInstance(characters, dict)
            
            # Restore original methods
            self.csm.narrator.process_narration = original_narrator_process
            self.csm.character_server.generate_response = original_cs_gen_response
            self.csm.client_manager.send_to_client = original_cm_send_to_client
            self.csm.client_manager.get_clients_for_story_progression = original_cm_get_clients
            self.csm.db.get_character = original_db_get_char
            
            await self.csm.shutdown_async()
            
        asyncio.run(run_integration_test())


if __name__ == '__main__':
    # Run all tests with verbose output
    unittest.main(verbosity=2)