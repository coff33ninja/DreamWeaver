import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, mock_open

# Ensure correct import path for LLMEngine from CharacterClient/src/
from CharacterClient.src.llm_engine import LLMEngine, DEFAULT_CLIENT_MODEL_NAME
from CharacterClient.src.config import CLIENT_LLM_MODELS_PATH

# Mock PEFT and Transformers objects that are loaded by llm_engine
MockPeftModel = MagicMock()
MockAutoModelForCausalLM = MagicMock()
MockAutoTokenizer = MagicMock()
MockBitsAndBytesConfig = MagicMock()
MockPrepareModelForKbitTraining = MagicMock()
MockLoraConfig = MagicMock()
MockGetPeftModel = MagicMock()
MockDataset = MagicMock()
MockTrainer = MagicMock()
MockTrainingArguments = MagicMock()
MockTorch = MagicMock()


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mocks external libraries used by LLMEngine."""
    monkeypatch.setattr("CharacterClient.src.llm_engine.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("CharacterClient.src.llm_engine.AutoTokenizer", MockAutoTokenizer)
    monkeypatch.setattr("CharacterClient.src.llm_engine.BitsAndBytesConfig", MockBitsAndBytesConfig)
    monkeypatch.setattr("CharacterClient.src.llm_engine.prepare_model_for_kbit_training", MockPrepareModelForKbitTraining)
    monkeypatch.setattr("CharacterClient.src.llm_engine.PeftModel", MockPeftModel)
    monkeypatch.setattr("CharacterClient.src.llm_engine.LoraConfig", MockLoraConfig)
    monkeypatch.setattr("CharacterClient.src.llm_engine.get_peft_model", MockGetPeftModel)
    monkeypatch.setattr("CharacterClient.src.llm_engine.torch", MockTorch)
    monkeypatch.setattr("CharacterClient.src.llm_engine.Dataset", MockDataset) # For fine_tune_async
    monkeypatch.setattr("CharacterClient.src.llm_engine.Trainer", MockTrainer) # For fine_tune_async
    monkeypatch.setattr("CharacterClient.src.llm_engine.TrainingArguments", MockTrainingArguments) # For fine_tune_async

    # Reset mocks before each test using this fixture
    MockAutoModelForCausalLM.from_pretrained.reset_mock()
    MockAutoTokenizer.from_pretrained.reset_mock()
    MockBitsAndBytesConfig.reset_mock()
    MockPrepareModelForKbitTraining.reset_mock()
    MockPeftModel.from_pretrained.reset_mock()
    MockGetPeftModel.reset_mock()
    MockTorch.cuda.is_available.return_value = False # Default to no CUDA
    MockTorch.cuda.is_bf16_supported.return_value = False


@pytest.fixture
def llm_engine_no_init(mock_dependencies):
    """Provides an LLMEngine instance without calling _load_model_and_tokenizer."""
    with patch.object(LLMEngine, "_load_model_and_tokenizer", MagicMock()): # Prevent auto-loading
        engine = LLMEngine(model_name="test_model", Actor_id="test_actor")
        engine.is_initialized = False # Explicitly set for clarity in tests
    return engine


class TestLLMEngineInitialization:

    def test_init_paths_and_defaults(self, mock_dependencies, llm_engine_no_init):
        engine = llm_engine_no_init # Already created with _load_model_and_tokenizer patched

        assert engine.Actor_id == "test_actor"
        assert engine.model_name == "test_model"
        sane_model_name_for_path = "test_model".replace("/", "_")
        expected_adapters_path = os.path.join(CLIENT_LLM_MODELS_PATH, "adapters", "test_actor", sane_model_name_for_path)
        assert engine.adapters_path == expected_adapters_path
        assert engine.models_base_path == os.path.join(CLIENT_LLM_MODELS_PATH, "base_models")
        assert not engine.is_initialized

    def test_init_default_model_name(self, mock_dependencies):
        with patch.object(LLMEngine, "_load_model_and_tokenizer", MagicMock()):
            engine = LLMEngine(Actor_id="default_actor_test") # Use default model name
            assert engine.model_name == DEFAULT_CLIENT_MODEL_NAME

    @patch("CharacterClient.src.llm_engine.os.makedirs")
    def test_init_creates_directories(self, mock_makedirs, mock_dependencies, llm_engine_no_init):
        engine = llm_engine_no_init # Directories are created before _load_model_and_tokenizer

        # Check that os.makedirs was called for base_models and adapters_path
        # The engine instance used here is from llm_engine_no_init,
        # but the LLMEngine constructor itself calls os.makedirs.
        # To test this properly, we'd need to instantiate LLMEngine inside the test
        # or ensure the fixture correctly captures these calls.
        # For now, let's assume the fixture `llm_engine_no_init` uses an engine where these were called.
        # This test might be more robust if LLMEngine is instantiated directly here.

        # Re-instantiate to capture makedirs calls for this specific test
        with patch.object(LLMEngine, "_load_model_and_tokenizer", MagicMock()):
            current_engine = LLMEngine(model_name="another_model", Actor_id="another_actor")
            sane_model_name = "another_model".replace("/", "_")
            expected_adapters_path = os.path.join(CLIENT_LLM_MODELS_PATH, "adapters", "another_actor", sane_model_name)
            expected_base_models_path = os.path.join(CLIENT_LLM_MODELS_PATH, "base_models")

            calls = [
                patch.call(expected_base_models_path, exist_ok=True),
                patch.call(expected_adapters_path, exist_ok=True),
            ]
            mock_makedirs.assert_has_calls(calls, any_order=True)


    def test_load_model_and_tokenizer_success_no_cuda_no_adapter(self, mock_dependencies, llm_engine_no_init):
        engine = llm_engine_no_init
        MockTorch.cuda.is_available.return_value = False

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "[EOS]"
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        MockAutoModelForCausalLM.from_pretrained.return_value = mock_model_instance

        with patch("CharacterClient.src.llm_engine.os.path.exists", return_value=False): # No adapter config
            engine._load_model_and_tokenizer()

        MockAutoTokenizer.from_pretrained.assert_called_once_with(engine.model_name, cache_dir=engine.models_base_path, trust_remote_code=True)
        assert mock_tokenizer_instance.pad_token == "[EOS]"
        MockAutoModelForCausalLM.from_pretrained.assert_called_once_with(
            engine.model_name,
            quantization_config=None,
            device_map="auto",
            cache_dir=engine.models_base_path,
            trust_remote_code=True
        )
        MockPrepareModelForKbitTraining.assert_not_called() # No CUDA
        MockPeftModel.from_pretrained.assert_not_called() # No adapter
        mock_model_instance.eval.assert_called_once()
        assert engine.is_initialized
        assert engine.model == mock_model_instance
        assert engine.tokenizer == mock_tokenizer_instance

    def test_load_model_and_tokenizer_success_with_cuda_and_adapter(self, mock_dependencies, llm_engine_no_init):
        engine = llm_engine_no_init
        MockTorch.cuda.is_available.return_value = True
        MockTorch.cuda.is_bf16_supported.return_value = True # For bnb_4bit_compute_dtype

        mock_tokenizer_instance = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_base_model_instance = MagicMock()
        MockAutoModelForCausalLM.from_pretrained.return_value = mock_base_model_instance

        mock_peft_model_instance = MagicMock()
        MockPeftModel.from_pretrained.return_value = mock_peft_model_instance

        # Simulate prepare_model_for_kbit_training returning the model
        MockPrepareModelForKbitTraining.return_value = mock_base_model_instance

        with patch("CharacterClient.src.llm_engine.os.path.exists", return_value=True): # Adapter config exists
            engine._load_model_and_tokenizer()

        MockBitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=MockTorch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        MockAutoModelForCausalLM.from_pretrained.assert_called_once() # Args checked by bnb_config mock
        MockPrepareModelForKbitTraining.assert_called_once_with(mock_base_model_instance)
        MockPeftModel.from_pretrained.assert_called_once_with(mock_base_model_instance, engine.adapters_path)

        mock_peft_model_instance.eval.assert_called_once()
        assert engine.is_initialized
        assert engine.model == mock_peft_model_instance # Model should be the PeftModel

    def test_load_model_failure_sets_not_initialized(self, mock_dependencies, llm_engine_no_init):
        engine = llm_engine_no_init
        MockAutoTokenizer.from_pretrained.side_effect = Exception("Load failed")

        engine._load_model_and_tokenizer()

        assert not engine.is_initialized
        assert engine.model is None
        assert engine.tokenizer is None

@pytest.mark.asyncio
class TestLLMEngineGenerate:
    async def test_generate_success(self, mock_dependencies, llm_engine_no_init):
        engine = llm_engine_no_init
        engine.is_initialized = True
        engine.model = MagicMock()
        engine.tokenizer = MagicMock()

        # Mock tokenizer call within _blocking_generate_task
        mock_inputs = {"input_ids": MagicMock(shape=(-1, 5))} # Mock shape for slicing
        engine.tokenizer.return_value = mock_inputs
        engine.tokenizer.model_max_length = 2048
        engine.tokenizer.pad_token_id = 0
        engine.tokenizer.eos_token_id = 1

        # Mock model.generate call
        mock_output_sequences = MagicMock()
        engine.model.generate.return_value = mock_output_sequences

        # Mock tokenizer.decode call
        engine.tokenizer.decode.return_value = "Generated text from model"

        prompt = "Test prompt"
        max_tokens = 50

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            # Make to_thread execute the passed function immediately for testing
            async def side_effect_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)
            mock_to_thread.side_effect = side_effect_to_thread

            response = await engine.generate(prompt, max_new_tokens=max_tokens)

        assert response == "Generated text from model"
        engine.tokenizer.assert_called_once_with(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048 - max_tokens - 10)
        mock_inputs.to.assert_called_once_with(engine.model.device)
        engine.model.generate.assert_called_once_with(
            **mock_inputs,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            do_sample=True, top_k=40, top_p=0.9, temperature=0.8,
            pad_token_id=0,
            eos_token_id=1
        )
        # The slicing is `output_sequences[0][inputs["input_ids"].shape[-1]:]`
        engine.tokenizer.decode.assert_called_once_with(mock_output_sequences[0][mock_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    async def test_generate_not_initialized(self, llm_engine_no_init):
        engine = llm_engine_no_init # is_initialized is False by fixture
        response = await engine.generate("Test prompt")
        assert response == "[LLM_ERROR:NOT_INITIALIZED]"

    async def test_generate_model_generate_fails(self, mock_dependencies, llm_engine_no_init):
        engine = llm_engine_no_init
        engine.is_initialized = True
        engine.model = MagicMock()
        engine.tokenizer = MagicMock()
        engine.model.generate.side_effect = Exception("Generation failed")

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            async def side_effect_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs) # Execute immediately
            mock_to_thread.side_effect = side_effect_to_thread

            response = await engine.generate("Test prompt")
        assert response == "[LLM_ERROR:GENERATION_FAILED]"


# Added DataCollatorForLanguageModeling to imports for llm_engine
from CharacterClient.src.llm_engine import LLMEngine, DEFAULT_CLIENT_MODEL_NAME, JsonDataset
from CharacterClient.src.config import CLIENT_LLM_MODELS_PATH
from transformers import DataCollatorForLanguageModeling


# Mock PEFT and Transformers objects that are loaded by llm_engine
MockPeftModel = MagicMock()
# ... (other mocks remain the same)
# Add DataCollator mock if not already there (it's used by the SUT)
MockDataCollatorForLanguageModeling = MagicMock()


@pytest.fixture
def mock_dependencies(monkeypatch):
    # ... (existing monkeypatch setup)
    monkeypatch.setattr("CharacterClient.src.llm_engine.DataCollatorForLanguageModeling", MockDataCollatorForLanguageModeling)
    MockDataCollatorForLanguageModeling.reset_mock()
    # ... (rest of the mock resets)


# ... (other test classes remain the same)


@pytest.mark.asyncio
class TestLLMEngineFineTuneAsync:
    @patch("CharacterClient.src.llm_engine.os.makedirs")
    @patch("CharacterClient.src.llm_engine.os.listdir")
    @patch("CharacterClient.src.llm_engine.json.dump")
    @patch("CharacterClient.src.llm_engine.json.load")
    @patch("builtins.open", new_callable=mock_open)
    async def test_fine_tune_async_saves_data_and_trains(
        self, mock_file_open, mock_json_load, mock_json_dump, mock_listdir, mock_os_makedirs,
        mock_dependencies, llm_engine_no_init
    ):
        engine = llm_engine_no_init
        engine.is_initialized = True
        engine.model = MockPeftModel() # So isinstance(engine.model, PeftModel) is true
        engine.tokenizer = MagicMock()

        mock_trainer_instance = MockTrainer.return_value

        # Simulate existing data and then new data
        mock_listdir.return_value = ["sample_0.json"]
        mock_json_load.return_value = {"input": "old input", "output": "old output"}

        training_data = {"input": "new input", "output": "new output"}
        actor_id = "test_actor_ft"

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            # Make to_thread execute the passed function immediately for testing
            async def side_effect_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)
            mock_to_thread.side_effect = side_effect_to_thread

            await engine.fine_tune_async(training_data, Actor_id_override=actor_id)

        # Verify data saving
        data_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "training_data_local", actor_id)
        mock_os_makedirs.assert_any_call(data_dir, exist_ok=True)
        # Filename for saving is random, so check json.dump call
        mock_json_dump.assert_called_once_with(training_data, mock_file_open.return_value.__enter__.return_value, ensure_ascii=False, indent=4)

        # Verify training
        MockDataset.from_list.assert_called_once() # Called with combined old and new data
        actual_dataset_input = MockDataset.from_list.call_args[0][0]
        assert len(actual_dataset_input) == 2 # old + new
        assert {"text": f"{training_data['input']}{engine.tokenizer.eos_token}{training_data['output']}{engine.tokenizer.eos_token}"} in actual_dataset_input
        
        MockTrainingArguments.assert_called_once()
        MockTrainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once_with(engine.adapters_path)
        
        engine.model.train.assert_called_once()
        engine.model.eval.assert_called_once() # Called after training


class TestLLMEngineSaveAdaptersAsync:
    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_save_adapters_async_peft_model(self, mock_to_thread, mock_dependencies, llm_engine_no_init):
        engine = llm_engine_no_init
        engine.is_initialized = True
        engine.model = MockPeftModel() # Make it an instance of the mocked PeftModel
        
        async def side_effect_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)
        mock_to_thread.side_effect = side_effect_to_thread

        await engine.save_adapters_async()
        
        engine.model.save_pretrained.assert_called_once_with(engine.adapters_path)

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_save_adapters_async_not_peft_model(self, mock_to_thread, mock_dependencies, llm_engine_no_init):
        engine = llm_engine_no_init
        engine.is_initialized = True
        engine.model = MagicMock() # Not a PeftModel instance

        await engine.save_adapters_async()
        
        engine.model.save_pretrained.assert_not_called()
        mock_to_thread.assert_not_called() # Should not even try to run in thread

    async def test_save_adapters_async_not_initialized(self, llm_engine_no_init):
        engine = llm_engine_no_init # is_initialized = False
        engine.model = MockPeftModel()

        await engine.save_adapters_async()
        
        engine.model.save_pretrained.assert_not_called()

# Configure logging for pytest capture if needed
logging.basicConfig(level=logging.DEBUG)
```
