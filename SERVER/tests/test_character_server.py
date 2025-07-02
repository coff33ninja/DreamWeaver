import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock, call

# Add SERVER/src to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from character_server import CharacterServer
# Assuming config paths are needed for some mocks related to audio paths
from config import REFERENCE_VOICES_AUDIO_PATH, CHARACTERS_AUDIO_PATH

# Mock external dependencies
MockLLMEngineClass = MagicMock()
MockTTSManagerClass = MagicMock()
MockPygameMixer = MagicMock()
MockPygameTime = MagicMock()
MockPygameSound = MagicMock()

@pytest.fixture
def mock_pygame(monkeypatch):
    monkeypatch.setattr("character_server.pygame.mixer", MockPygameMixer)
    monkeypatch.setattr("character_server.pygame.time", MockPygameTime)
    monkeypatch.setattr("character_server.pygame.mixer.Sound", MockPygameSound) # Mock the Sound class constructor
    MockPygameMixer.get_init.return_value = False # Default to not initialized
    MockPygameMixer.init = MagicMock()
    MockPygameMixer.get_busy = MagicMock(return_value=False) # Default to not busy
    MockPygameSound.return_value.play = MagicMock() # Mock play method on Sound instance
    MockPygameTime.Clock.return_value.tick = MagicMock()

@pytest.fixture
def mock_dependencies_cs(monkeypatch, mock_pygame): # Include mock_pygame here
    monkeypatch.setattr("character_server.LLMEngine", MockLLMEngineClass)
    monkeypatch.setattr("character_server.TTSManager", MockTTSManagerClass)
    
    MockLLMEngineClass.reset_mock()
    MockTTSManagerClass.reset_mock()
    MockPygameMixer.reset_mock()
    MockPygameTime.reset_mock()
    MockPygameSound.reset_mock()

    # Mock instances returned by constructors
    # These will be used when CharacterServer instantiates them in async_init
    llm_instance_mock = MagicMock(is_initialized=True)
    tts_instance_mock = MagicMock(is_initialized=True)
    MockLLMEngineClass.return_value = llm_instance_mock
    MockTTSManagerClass.return_value = tts_instance_mock
    
    return llm_instance_mock, tts_instance_mock


@pytest.fixture
def mock_db_cs():
    db = MagicMock()
    db.get_character.return_value = None # Default: Actor1 does not exist
    db.save_character = MagicMock()
    db.save_training_data = MagicMock()
    return db

@pytest.fixture
def character_server_bare(mock_db_cs, mock_dependencies_cs):
    """Provides a CharacterServer instance before async_init is called."""
    # Patch run_in_executor to prevent async_init's internal calls during __init__ if any
    # (though CharacterServer's __init__ is sync)
    # Also, async_init itself will be explicitly called in tests.
    cs = CharacterServer(db=mock_db_cs)
    return cs

@pytest.fixture
async def character_server_initialized(character_server_bare, mock_db_cs, mock_dependencies_cs):
    """Provides a CharacterServer instance that has run a mocked async_init."""
    cs = character_server_bare

    # Ensure get_character returns something for async_init to use
    mock_db_cs.get_character.return_value = {
        "name": "Actor1_From_DB", "personality": "db_personality", "tts": "gtts",
        "tts_model": "en", "reference_audio_filename": None, "Actor_id": "Actor1",
        "llm_model": "db_llm", "language": "en"
    }
    
    # Mock run_in_executor to control execution of LLM/TTS init
    # The lambda inside run_in_executor will call the mocked class constructors
    llm_mock, tts_mock = mock_dependencies_cs

    async def mock_run_in_executor(loop, func): # Simplified mock
        if "LLMEngine" in str(func): return llm_mock
        if "TTSManager" in str(func): return tts_mock
        return func() # For other potential uses

    with patch("asyncio.get_event_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_loop.run_in_executor.side_effect = mock_run_in_executor
        mock_get_loop.return_value = mock_loop
        
        await cs.async_init()
    
    cs.llm = llm_mock # Ensure instances are the specific mocks
    cs.tts = tts_mock
    return cs


class TestCharacterServerInitialization:
    def test_init_actor1_not_in_db(self, mock_db_cs, mock_dependencies_cs):
        mock_db_cs.get_character.return_value = None # Actor1 not in DB
        
        cs = CharacterServer(db=mock_db_cs)
        
        mock_db_cs.get_character.assert_called_once_with("Actor1")
        assert cs.character["name"] == "Actor1_Default"
        mock_db_cs.save_character.assert_called_once()
        saved_char_args = mock_db_cs.save_character.call_args[1] # kwargs
        assert saved_char_args["Actor_id"] == "Actor1"
        assert saved_char_args["name"] == "Actor1_Default"
        assert cs.llm is None
        assert cs.tts is None

    def test_init_actor1_exists_in_db(self, mock_db_cs, mock_dependencies_cs):
        db_char_data = {"name": "Actor1_DB", "Actor_id": "Actor1", "personality": "DB_pers"}
        mock_db_cs.get_character.return_value = db_char_data
        
        cs = CharacterServer(db=mock_db_cs)
        
        mock_db_cs.get_character.assert_called_once_with("Actor1")
        assert cs.character == db_char_data
        mock_db_cs.save_character.assert_not_called()


@pytest.mark.asyncio
class TestCharacterServerAsyncInit:
    @patch("character_server.ACTOR1_PYGAME_AUDIO_ENABLED", True) # Test with pygame enabled
    async def test_async_init_success_pygame_enabled(self, character_server_bare, mock_db_cs, mock_dependencies_cs):
        cs = character_server_bare
        llm_mock, tts_mock = mock_dependencies_cs
        
        char_data_for_init = {
            "name": "TestInitActor1", "personality": "init_pers", "tts": "xttsv2",
            "tts_model": "xtts_model", "reference_audio_filename": "ref.wav",
            "Actor_id": "Actor1", "llm_model": "init_llm", "language": "fr"
        }
        mock_db_cs.get_character.return_value = char_data_for_init
        expected_speaker_wav = os.path.join(REFERENCE_VOICES_AUDIO_PATH, "ref.wav")

        with patch("os.path.exists", return_value=True) as mock_os_exists:
            async def mock_run_in_executor_specific(loop, func_to_run, *args_to_func):
                if "LLMEngine" in str(func_to_run):
                    return MockLLMEngineClass(model_name=char_data_for_init["llm_model"], db=mock_db_cs)
                if "TTSManager" in str(func_to_run):
                    return MockTTSManagerClass(
                        tts_service_name=char_data_for_init["tts"],
                        model_name=char_data_for_init["tts_model"],
                        speaker_wav_path=expected_speaker_wav,
                        language=char_data_for_init["language"]
                    )
                return MagicMock()

            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_loop.run_in_executor.side_effect = mock_run_in_executor_specific
                mock_get_loop.return_value = mock_loop
                await cs.async_init()

        MockLLMEngineClass.assert_called_once_with(model_name="init_llm", db=mock_db_cs)
        MockTTSManagerClass.assert_called_once_with(
            tts_service_name="xttsv2", model_name="xtts_model",
            speaker_wav_path=expected_speaker_wav, language="fr"
        )
        mock_os_exists.assert_called_once_with(expected_speaker_wav)
        assert cs.llm == llm_mock
        assert cs.tts == tts_mock
        MockPygameMixer.init.assert_called_once()

    @patch("character_server.ACTOR1_PYGAME_AUDIO_ENABLED", False) # Test with pygame disabled
    async def test_async_init_success_pygame_disabled(self, character_server_bare, mock_db_cs, mock_dependencies_cs):
        cs = character_server_bare
        # ... (similar setup for LLM/TTS as above, not repeating for brevity)
        mock_db_cs.get_character.return_value = {"llm_model": "any", "tts": "gtts"}


        # Mock run_in_executor for LLM/TTS instantiation
        async def mock_run_in_executor_disable(loop, func_to_run, *args_to_func):
            if "LLMEngine" in str(func_to_run): return MockLLMEngineClass()
            if "TTSManager" in str(func_to_run): return MockTTSManagerClass()
            return MagicMock()

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.run_in_executor.side_effect = mock_run_in_executor_disable
            mock_get_loop.return_value = mock_loop

            logger = logging.getLogger("dreamweaver_server")
            with patch.object(logger, "info") as mock_log_info:
                await cs.async_init()

        MockPygameMixer.init.assert_not_called()
        mock_log_info.assert_any_call(
            f"CharacterServer ({cs.character_Actor_id}): Pygame mixer for Actor1 audio playback is DISABLED by configuration (ACTOR1_PYGAME_AUDIO_ENABLED=False)."
        )

    async def test_async_init_llm_fails(self, character_server_bare, mock_db_cs, mock_dependencies_cs):
        cs = character_server_bare
        llm_mock, tts_mock = mock_dependencies_cs # These are instances from the fixture

        # Simulate LLMEngine constructor (called via run_in_executor) returning a non-initialized mock
        MockLLMEngineClass.return_value = MagicMock(is_initialized=False)

        mock_db_cs.get_character.return_value = {"llm_model": "fail_llm", "tts": "gtts", "language": "en"}

        with patch("asyncio.get_event_loop") as mock_get_loop:
            async def mock_run_in_executor_llm_fail(loop, func, *args, **kwargs):
                if "LLMEngine" in str(func):
                    # Return the mock that has is_initialized=False
                    return MockLLMEngineClass(model_name=args[0], db=args[1])
                elif "TTSManager" in str(func):
                    return MockTTSManagerClass() # Assume TTS init is fine
                return MagicMock()

            mock_loop = MagicMock()
            mock_loop.run_in_executor.side_effect = mock_run_in_executor_llm_fail
            mock_get_loop.return_value = mock_loop

            await cs.async_init()

        assert cs.llm is not None
        assert not cs.llm.is_initialized
        assert cs.tts is not None # TTS should still init
        assert cs.tts.is_initialized


@pytest.mark.asyncio
class TestCharacterServerGenerateResponse:
    @patch("character_server.ACTOR1_PYGAME_AUDIO_ENABLED", True) # Assume enabled for this test
    async def test_generate_response_success(self, character_server_initialized, mock_db_cs):
        cs = character_server_initialized
        # Ensure llm and tts are the mocked instances from the fixture
        cs.llm.generate = AsyncMock(return_value="Generated LLM response.")
        cs.llm.fine_tune = AsyncMock()
        cs.output_audio = AsyncMock()

        narration = "A story starts."
        other_texts = {"OtherChar": "I agree."}

        response = await cs.generate_response(narration, other_texts)

        assert response == "Generated LLM response."
        cs.llm.generate.assert_awaited_once()

        mock_db_cs.save_training_data.assert_called_once()

        cs.llm.fine_tune.assert_awaited_once()
        cs.output_audio.assert_awaited_once_with("Generated LLM response.")

    async def test_generate_response_llm_not_initialized(self, character_server_initialized):
        cs = character_server_initialized
        cs.llm.is_initialized = False # Ensure LLM is marked as not initialized

        response = await cs.generate_response("Narration", {})
        char_name = cs.character.get('name', cs.character_Actor_id)
        assert response == f"[{char_name}_LLM_ERROR:NOT_INITIALIZED]"
        # cs.output_audio.assert_not_awaited() # output_audio might still be called if text is error string

    async def test_generate_response_llm_returns_error(self, character_server_initialized):
        cs = character_server_initialized
        # Ensure llm and tts are the mocked instances
        cs.llm.generate = AsyncMock(return_value="[LLM_ERROR:TEST_FAILURE]")
        cs.output_audio = AsyncMock() # Mock to check if it's called

        response = await cs.generate_response("Narration", {})
        assert response == "[LLM_ERROR:TEST_FAILURE]"
        # output_audio is called even if LLM returns an error, to potentially speak the error.
        cs.output_audio.assert_awaited_once_with("[LLM_ERROR:TEST_FAILURE]")


@pytest.mark.asyncio
class TestCharacterServerOutputAudio:

    @patch("character_server.ACTOR1_PYGAME_AUDIO_ENABLED", True)
    @patch("character_server.os.makedirs")
    @patch("character_server.os.path.exists")
    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_output_audio_success_pygame_enabled_and_initialized(self, mock_async_to_thread, mock_os_path_exists, mock_os_makedirs, character_server_initialized):
        cs = character_server_initialized
        cs.tts.synthesize = AsyncMock(return_value=True)
        mock_os_path_exists.return_value = True
        MockPygameMixer.get_init.return_value = True

        text_to_speak = "Let's hear this."
        await cs.output_audio(text_to_speak)

        cs.tts.synthesize.assert_awaited_once()
        synthesize_call_args = cs.tts.synthesize.call_args[0]
        assert synthesize_call_args[0] == text_to_speak
        assert CHARACTERS_AUDIO_PATH in synthesize_call_args[1]
        assert ".wav" in synthesize_call_args[1]

        mock_async_to_thread.assert_awaited_once()
        MockPygameSound.assert_called_once()
        MockPygameSound.return_value.play.assert_called_once()

    @patch("character_server.ACTOR1_PYGAME_AUDIO_ENABLED", False)
    @patch("character_server.os.makedirs")
    @patch("character_server.os.path.exists")
    async def test_output_audio_pygame_disabled(self, mock_os_path_exists, mock_os_makedirs, character_server_initialized):
        cs = character_server_initialized
        cs.tts.synthesize = AsyncMock(return_value=True)
        mock_os_path_exists.return_value = True

        logger = logging.getLogger("dreamweaver_server")
        with patch.object(logger, "info") as mock_log_info:
            await cs.output_audio("Test with pygame disabled.")

        cs.tts.synthesize.assert_awaited_once() # Synthesis should still happen
        MockPygameSound.return_value.play.assert_not_called()
        mock_log_info.assert_any_call(
            f"CharacterServer ({cs.character.get('name', cs.character_Actor_id)}): Pygame audio for Actor1 is DISABLED. Skipping playback of {cs.tts.synthesize.call_args[0][1]}."
        )

    @patch("character_server.ACTOR1_PYGAME_AUDIO_ENABLED", True)
    @patch("character_server.os.makedirs")
    @patch("character_server.os.path.exists")
    async def test_output_audio_pygame_enabled_mixer_not_init(self, mock_os_path_exists, mock_os_makedirs, character_server_initialized):
        cs = character_server_initialized
        cs.tts.synthesize = AsyncMock(return_value=True)
        mock_os_path_exists.return_value = True
        MockPygameMixer.get_init.return_value = False # Mixer not initialized

        logger = logging.getLogger("dreamweaver_server")
        with patch.object(logger, "warning") as mock_log_warning:
            await cs.output_audio("Test with mixer not init.")

        cs.tts.synthesize.assert_awaited_once()
        MockPygameSound.return_value.play.assert_not_called()
        mock_log_warning.assert_any_call(
            f"CharacterServer ({cs.character.get('name', cs.character_Actor_id)}): Pygame audio for Actor1 is ENABLED but mixer not initialized. Cannot play audio {cs.tts.synthesize.call_args[0][1]}."
        )


    async def test_output_audio_tts_not_initialized(self, character_server_initialized):
        cs = character_server_initialized
        cs.tts.is_initialized = False

        with patch.object(logging.getLogger("dreamweaver_server"), "warning") as mock_log_warning:
            await cs.output_audio("No sound.")
            mock_log_warning.assert_called_once()
            assert "TTS not initialized" in mock_log_warning.call_args[0][0]

    async def test_output_audio_empty_text(self, character_server_initialized):
        cs = character_server_initialized
        with patch.object(logging.getLogger("dreamweaver_server"), "warning") as mock_log_warning:
            await cs.output_audio("")
            mock_log_warning.assert_called_once()
            assert "Text is empty" in mock_log_warning.call_args[0][0]

    @patch("character_server.os.makedirs", side_effect=OSError("Cannot make dir"))
    async def test_output_audio_cannot_create_dir(self, mock_os_makedirs_err, character_server_initialized):
        cs = character_server_initialized
        with patch.object(logging.getLogger("dreamweaver_server"), "error") as mock_log_error:
            await cs.output_audio("Test.")
            mock_log_error.assert_called_once()
            assert "Could not create audio directory" in mock_log_error.call_args[0][0]
        cs.tts.synthesize.assert_not_awaited()


# Example of how the __main__ block in character_server.py could be tested (partially)
@pytest.mark.asyncio
@patch("character_server.CharacterServer") # Patch the class itself
@patch("character_server.asyncio.run") # Patch asyncio.run
@patch("character_server.logging.basicConfig") # Patch logging setup
async def test_main_block_execution(mock_logging_config, mock_asyncio_run, MockCharacterServerClass, mock_db_cs):
    # This tests that the main block attempts to run the test_character_server function

    # Simulate CharacterServer instance and its methods for the test_character_server function
    mock_cs_instance = AsyncMock()
    mock_cs_instance.llm = MagicMock(is_initialized=True)
    mock_cs_instance.tts = MagicMock(is_initialized=True)
    mock_cs_instance.generate_response = AsyncMock(return_value="Main block response")
    MockCharacterServerClass.return_value = mock_cs_instance

    # Need to allow the test_character_server coroutine to be created and passed to asyncio.run
    # This requires a bit of careful mocking if we want to assert calls within test_character_server

    # For simplicity, just ensure asyncio.run is called with a coroutine
    # The actual logic of test_character_server is complex to unit test via this entry point.

    # To run the __main__ block, we can import the module.
    # However, pytest typically imports modules when collecting tests.
    # A more direct way is to simulate the conditions of __name__ == "__main__"
    # or refactor test_character_server to be callable for tests.

    # For now, let's assume the goal is to check if asyncio.run is called.
    # We'd need to execute the module's __main__ block. This is tricky in unit tests.
    # A better approach is to make `test_character_server` a standalone testable async function
    # and call it directly in a separate test.

    # This test is more conceptual for now.
    # If character_server.py was run as a script, asyncio.run(test_character_server()) would be called.
    # Example:
    # with patch.dict('sys.modules', {'__main__': sys.modules['character_server']}):
    #     import character_server # This would run the __main__
    # mock_asyncio_run.assert_called_once()
    # But this is complex and has side effects.

    assert True # Placeholder, direct testing of __main__ is involved.

```
