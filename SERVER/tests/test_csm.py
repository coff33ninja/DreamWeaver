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
        """Set up test fixtures."""
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def test_csm_initialization(self):
        """Test CSM initialization creates all required components."""
        self.assertIsNotNone(self.csm)
        self.assertTrue(hasattr(self.csm, 'narrator'))
        self.assertTrue(hasattr(self.csm, 'character_server'))
        self.assertTrue(hasattr(self.csm, 'client_manager'))
        self.assertTrue(hasattr(self.csm, 'db'))
    
    def test_csm_multiple_instances(self):
        """Test that multiple CSM instances are independent."""
        csm1 = CSM()
        csm2 = CSM()
        self.assertIsNot(csm1, csm2)


class TestCSMProcessStory(unittest.TestCase):
    """Test the main process_story functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
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
        """Clean up test fixtures."""
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
        """Test successful story processing with all components working."""
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
        """Test process_story when narrator fails."""
        mock_narrator.process_narration = AsyncMock(side_effect=Exception("Narrator error"))
        
        async def run_test():
            with self.assertRaises(Exception) as context:
                await self.csm.process_story(self.test_audio_file)
            self.assertIn("Narrator error", str(context.exception))
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    def test_process_story_character_server_failure(self, mock_cs, mock_narrator):
        """Test process_story when character server fails."""
        mock_narrator.process_narration = AsyncMock(return_value=self.dummy_narration)
        mock_cs.generate_response = AsyncMock(side_effect=Exception("Character server error"))
        
        async def run_test():
            with self.assertRaises(Exception) as context:
                await self.csm.process_story(self.test_audio_file)
            self.assertIn("Character server error", str(context.exception))
        
        asyncio.run(run_test())
    
    def test_process_story_invalid_audio_file(self):
        """Test process_story with non-existent audio file."""
        async def run_test():
            with self.assertRaises((FileNotFoundError, Exception)):
                await self.csm.process_story("nonexistent_file.wav")
        
        asyncio.run(run_test())
    
    def test_process_story_invalid_input_types(self):
        """Test process_story with invalid input types."""
        invalid_inputs = [None, 123, [], {}, ""]
        
        async def run_test():
            for invalid_input in invalid_inputs:
                with self.assertRaises((TypeError, ValueError, Exception)):
                    await self.csm.process_story(invalid_input)
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    def test_process_story_chaos_level_variations(self, mock_cm, mock_cs, mock_narrator):
        """Test process_story with different chaos levels."""
        mock_narrator.process_narration = AsyncMock(return_value=self.dummy_narration)
        mock_cs.generate_response = AsyncMock(return_value="Character response")
        mock_cm.get_clients_for_story_progression = Mock(return_value=[])
        
        chaos_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        async def run_test():
            for chaos_level in chaos_levels:
                narration, characters = await self.csm.process_story(
                    self.test_audio_file, 
                    chaos_level=chaos_level
                )
                self.assertIsNotNone(narration)
                self.assertIsInstance(characters, dict)
        
        asyncio.run(run_test())
    
    def test_process_story_invalid_chaos_levels(self):
        """Test process_story with invalid chaos levels."""
        invalid_chaos_levels = [-0.1, 1.1, 2.0, "invalid", None, float('inf')]
        
        async def run_test():
            for chaos_level in invalid_chaos_levels:
                with self.assertRaises((ValueError, TypeError)):
                    await self.csm.process_story(self.test_audio_file, chaos_level=chaos_level)
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    @patch('csm.CSM.db')
    def test_process_story_no_clients(self, mock_db, mock_cm, mock_cs, mock_narrator):
        """Test process_story when no clients are available."""
        mock_narrator.process_narration = AsyncMock(return_value=self.dummy_narration)
        mock_cs.generate_response = AsyncMock(return_value="Character response")
        mock_cm.get_clients_for_story_progression = Mock(return_value=[])
        
        async def run_test():
            narration, characters = await self.csm.process_story(self.test_audio_file)
            self.assertIsNotNone(narration)
            self.assertIsInstance(characters, dict)
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    @patch('csm.CSM.db')
    def test_process_story_multiple_clients(self, mock_db, mock_cm, mock_cs, mock_narrator):
        """Test process_story with multiple clients."""
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
            narration, characters = await self.csm.process_story(self.test_audio_file)
            self.assertIsNotNone(narration)
            self.assertIsInstance(characters, dict)
            self.assertEqual(len(characters), 3)
        
        asyncio.run(run_test())


class TestCSMErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            asyncio.run(self.csm.shutdown_async())
        except:
            pass
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    @patch('csm.CSM.db')
    def test_database_connection_failure(self, mock_db, mock_cm, mock_cs, mock_narrator):
        """Test handling of database connection failures."""
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
            with self.assertRaises(Exception) as context:
                await self.csm.process_story("test.wav")
            self.assertIn("Database connection failed", str(context.exception))
        
        asyncio.run(run_test())
    
    @patch('csm.CSM.narrator')
    @patch('csm.CSM.character_server')
    @patch('csm.CSM.client_manager')
    @patch('csm.CSM.db')
    def test_client_communication_failure(self, mock_db, mock_cm, mock_cs, mock_narrator):
        """Test handling of client communication failures."""
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
            try:
                narration, characters = await self.csm.process_story("test.wav")
                # Might return partial results or raise exception
                self.assertIsNotNone(narration)
            except Exception as e:
                self.assertIn("Client unreachable", str(e))
        
        asyncio.run(run_test())
    
    def test_concurrent_process_story_calls(self):
        """Test concurrent calls to process_story."""
        async def run_test():
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
        """Set up test fixtures."""
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def test_shutdown_async(self):
        """Test CSM async shutdown."""
        async def run_test():
            # Mock shutdown methods if they exist
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
        """Test that shutdown can be called multiple times safely."""
        async def run_test():
            await self.csm.shutdown_async()
            # Should not raise exception on second call
            await self.csm.shutdown_async()
        
        asyncio.run(run_test())


class TestCSMEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            asyncio.run(self.csm.shutdown_async())
        except:
            pass
    
    def test_very_long_audio_filename(self):
        """Test with very long audio filenames."""
        long_filename = "a" * 1000 + ".wav"
        
        async def run_test():
            with self.assertRaises((OSError, Exception)):
                await self.csm.process_story(long_filename)
        
        asyncio.run(run_test())
    
    def test_special_characters_in_filename(self):
        """Test audio filenames with special characters."""
        special_filenames = [
            "file with spaces.wav",
            "file-with-dashes.wav",
            "file_with_underscores.wav",
            "file.with.dots.wav"
        ]
        
        async def run_test():
            for filename in special_filenames:
                try:
                    await self.csm.process_story(filename)
                except (FileNotFoundError, Exception):
                    pass  # Expected for non-existent files
        
        asyncio.run(run_test())
    
    def test_unicode_filenames(self):
        """Test with Unicode characters in filenames."""
        unicode_filenames = [
            "测试.wav",
            "tëst.wav",
            "тест.wav"
        ]
        
        async def run_test():
            for filename in unicode_filenames:
                try:
                    await self.csm.process_story(filename)
                except (FileNotFoundError, Exception):
                    pass  # Expected for non-existent files
        
        asyncio.run(run_test())


class TestCSMPerformance(unittest.TestCase):
    """Performance and timeout tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            asyncio.run(self.csm.shutdown_async())
        except:
            pass
    
    @patch('csm.CSM.narrator')
    def test_timeout_handling(self, mock_narrator):
        """Test handling of operation timeouts."""
        async def slow_process(audio_file):
            await asyncio.sleep(10)  # Simulate very slow processing
            return {"text": "Slow result", "audio_path": audio_file, "speaker": "Narrator"}
        
        mock_narrator.process_narration = slow_process
        
        async def run_test():
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
        """Set up integration test."""
        if CSM is None:
            self.skipTest("CSM module not available")
        self.csm = CSM()
        self.dummy_audio_file = "csm_test_narration.wav"
    
    def tearDown(self):
        """Clean up integration test."""
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
        """Integration test for CSM process_story (enhanced from original)."""
        async def run_integration_test():
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
                return {"text": "This is a test narration from mock.", "audio_path": audio_filepath, "speaker": "Narrator"}
            self.csm.narrator.process_narration = mock_narrator_process
            
            original_cs_gen_response = self.csm.character_server.generate_response
            async def mock_cs_gen_response(narration, other_texts):
                return "Actor1 says hello asynchronously!"
            self.csm.character_server.generate_response = mock_cs_gen_response
            
            original_cm_send_to_client = self.csm.client_manager.send_to_client
            async def mock_cm_send_to_client(client_actor_id, client_ip, client_port, narration, character_texts):
                return f"{client_actor_id} says hi via async mock!"
            self.csm.client_manager.send_to_client = mock_cm_send_to_client
            
            original_cm_get_clients = self.csm.client_manager.get_clients_for_story_progression
            def mock_cm_get_clients():
                return [{"Actor_id": "Actor_TestClient", "ip_address": "127.0.0.1", "client_port": 8001}]
            self.csm.client_manager.get_clients_for_story_progression = mock_cm_get_clients
            
            original_db_get_char = self.csm.db.get_character
            def mock_db_get_char(Actor_id):
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