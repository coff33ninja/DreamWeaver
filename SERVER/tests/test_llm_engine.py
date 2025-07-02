import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, mock_open, AsyncMock

# Ensure correct import path for LLMEngine from SERVER/src/
from SERVER.src.llm_engine import LLMEngine
from SERVER.src.config import ADAPTERS_PATH, MODELS_PATH  # Server specific config paths

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
MockDataCollatorForLanguageModeling = MagicMock()


@pytest.fixture(autouse=True) # Apply to all tests in this module
def reset_global_mocks():
    """Resets global mocks before each test."""
    MockAutoModelForCausalLM.from_pretrained.reset_mock()
    MockAutoTokenizer.from_pretrained.reset_mock()
    MockBitsAndBytesConfig.reset_mock()
    MockPrepareModelForKbitTraining.reset_mock()
    MockPeftModel.from_pretrained.reset_mock()
    MockPeftModel.save_pretrained.reset_mock() # Also reset save_pretrained on the class
    MockGetPeftModel.reset_mock()
    MockLoraConfig.reset_mock()
    MockDataCollatorForLanguageModeling.reset_mock()
    MockTrainer.reset_mock()
    MockTrainingArguments.reset_mock()
    MockDataset.from_list.reset_mock()
    MockTorch.cuda.is_available.return_value = False
    MockTorch.cuda.is_bf16_supported.return_value = False

    # Configure return value for get_peft_model to be a mock that has methods like eval, train, save_pretrained
    peft_model_instance = MagicMock(spec=MockPeftModel)
    peft_model_instance.save_pretrained = MagicMock()
    peft_model_instance.eval = MagicMock()
    peft_model_instance.train = MagicMock()
    peft_model_instance.device = "cpu" # Mock device
    MockGetPeftModel.return_value = peft_model_instance

    # Configure AutoModelForCausalLM.from_pretrained to return a mock with necessary attributes
    base_model_instance = MagicMock()
    base_model_instance.device = "cpu" # Mock device
    MockAutoModelForCausalLM.from_pretrained.return_value = base_model_instance


@pytest.fixture
def mock_server_dependencies(monkeypatch):
    """Mocks external libraries used by server's LLMEngine."""
    monkeypatch.setattr("SERVER.src.llm_engine.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("SERVER.src.llm_engine.AutoTokenizer", MockAutoTokenizer)
    monkeypatch.setattr("SERVER.src.llm_engine.BitsAndBytesConfig", MockBitsAndBytesConfig)
    monkeypatch.setattr("SERVER.src.llm_engine.prepare_model_for_kbit_training", MockPrepareModelForKbitTraining)
    monkeypatch.setattr("SERVER.src.llm_engine.PeftModel", MockPeftModel)
    monkeypatch.setattr("SERVER.src.llm_engine.LoraConfig", MockLoraConfig)
    monkeypatch.setattr("SERVER.src.llm_engine.get_peft_model", MockGetPeftModel)
    monkeypatch.setattr("SERVER.src.llm_engine.torch", MockTorch)
    monkeypatch.setattr("SERVER.src.llm_engine.Dataset", MockDataset)
    monkeypatch.setattr("SERVER.src.llm_engine.Trainer", MockTrainer)
    monkeypatch.setattr("SERVER.src.llm_engine.TrainingArguments", MockTrainingArguments)
    monkeypatch.setattr("SERVER.src.llm_engine.DataCollatorForLanguageModeling", MockDataCollatorForLanguageModeling)


@pytest.fixture
def mock_db():
    """Fixture for a mocked database object."""
    db = MagicMock()
    db.get_training_data_for_Actor.return_value = []
    return db

@pytest.fixture
def server_llm_engine_no_init_load(mock_server_dependencies, mock_db):
    with patch.object(LLMEngine, "_load_model", MagicMock()):
        engine = LLMEngine(model_name="server_test_model", db=mock_db)
        engine.is_initialized = False
    return engine


class TestServerLLMEngineInitialization:

    def test_init_paths_and_defaults(self, server_llm_engine_no_init_load, mock_db):
        engine = server_llm_engine_no_init_load
        assert engine.db == mock_db
        assert engine.model_name == "server_test_model"
        sane_model_name = "server_test_model".replace("/", "_")
        expected_adapter_path = os.path.join(ADAPTERS_PATH, sane_model_name, "Actor1")
        assert engine.adapter_path == expected_adapter_path
        assert engine.base_model_cache_path == os.path.join(MODELS_PATH, "llm_base_models")
        assert not engine.is_initialized
        assert isinstance(engine._load_model, MagicMock)

    @patch("SERVER.src.llm_engine.os.makedirs")
    def test_init_creates_directories(self, mock_makedirs, mock_server_dependencies, mock_db):
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_event_loop = MagicMock()
            mock_event_loop.run_in_executor = MagicMock()
            mock_get_loop.return_value = mock_event_loop
            engine = LLMEngine(model_name="dir_test_model", db=mock_db)
            sane_model_name = "dir_test_model".replace("/", "_")
            expected_adapter_path = os.path.join(ADAPTERS_PATH, sane_model_name, "Actor1")
            expected_base_model_cache_path = os.path.join(MODELS_PATH, "llm_base_models")
            calls = [
                patch.call(expected_adapter_path, exist_ok=True),
                patch.call(expected_base_model_cache_path, exist_ok=True),
            ]
            mock_makedirs.assert_has_calls(calls, any_order=True)
            mock_event_loop.run_in_executor.assert_called_once_with(None, engine._load_model)

    def test_load_model_no_cuda_no_adapter_creates_peft(self, server_llm_engine_no_init_load):
        engine = server_llm_engine_no_init_load
        engine._load_model = LLMEngine._load_model.__get__(engine, LLMEngine)
        MockTorch.cuda.is_available.return_value = False
        mock_tokenizer_instance = MagicMock(pad_token=None, eos_token="[EOS]")
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_instance = MockAutoModelForCausalLM.from_pretrained.return_value # Already set by fixture

        with patch("SERVER.src.llm_engine.os.path.exists", return_value=False): # No adapter
            engine._load_model()

        MockAutoTokenizer.from_pretrained.assert_called_once_with(engine.model_name, trust_remote_code=True, cache_dir=engine.base_model_cache_path)
        assert mock_tokenizer_instance.pad_token == "[EOS]"
        MockAutoModelForCausalLM.from_pretrained.assert_called_once_with(
            engine.model_name, quantization_config=None, device_map="auto",
            trust_remote_code=True, cache_dir=engine.base_model_cache_path
        )
        MockPrepareModelForKbitTraining.assert_not_called()
        MockPeftModel.from_pretrained.assert_not_called()
        MockLoraConfig.assert_called_once()
        MockGetPeftModel.assert_called_once_with(mock_model_instance, MockLoraConfig.return_value)
        MockGetPeftModel.return_value.eval.assert_called_once()
        assert engine.is_initialized
        assert engine.model == MockGetPeftModel.return_value
        assert engine.tokenizer == mock_tokenizer_instance

    def test_load_model_with_cuda_and_adapter(self, server_llm_engine_no_init_load):
        engine = server_llm_engine_no_init_load
        engine._load_model = LLMEngine._load_model.__get__(engine, LLMEngine)
        MockTorch.cuda.is_available.return_value = True
        MockTorch.cuda.is_bf16_supported.return_value = True
        mock_tokenizer_instance = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_base_model_instance = MockAutoModelForCausalLM.from_pretrained.return_value

        mock_loaded_peft_model_instance = MagicMock(spec=MockPeftModel)
        mock_loaded_peft_model_instance.eval = MagicMock()
        MockPeftModel.from_pretrained.return_value = mock_loaded_peft_model_instance
        MockPrepareModelForKbitTraining.return_value = mock_base_model_instance

        with patch("SERVER.src.llm_engine.os.path.exists", return_value=True): # Adapter exists
            engine._load_model()

        MockBitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=MockTorch.bfloat16, bnb_4bit_use_double_quant=False,
        )
        MockPrepareModelForKbitTraining.assert_called_once_with(mock_base_model_instance)
        MockPeftModel.from_pretrained.assert_called_once_with(mock_base_model_instance, engine.adapter_path)
        MockGetPeftModel.assert_not_called()
        mock_loaded_peft_model_instance.eval.assert_called_once()
        assert engine.is_initialized
        assert engine.model == mock_loaded_peft_model_instance

@pytest.mark.asyncio
class TestServerLLMEngineGenerate:
    async def test_generate_success(self, server_llm_engine_no_init_load):
        engine = server_llm_engine_no_init_load
        engine.is_initialized = True
        engine.model = MagicMock(device="cpu") # Mock model with device
        engine.tokenizer = MagicMock()
        mock_inputs = {"input_ids": MagicMock(shape=(-1, 5))}
        engine.tokenizer.return_value = mock_inputs
        mock_inputs.to = MagicMock(return_value=mock_inputs) # Mock .to(device)
        engine.tokenizer.model_max_length = 1024
        engine.tokenizer.pad_token_id = 1
        engine.tokenizer.eos_token_id = 2
        mock_output_sequences = MagicMock()
        engine.model.generate.return_value = mock_output_sequences
        engine.tokenizer.decode.return_value = "Server generated text"
        prompt = "Server prompt"
        max_tokens = 60

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            async def side_effect_to_thread(func, *args, **kwargs): return func(*args, **kwargs)
            mock_to_thread.side_effect = side_effect_to_thread
            response = await engine.generate(prompt, max_new_tokens=max_tokens)

        assert response == "Server generated text"
        engine.tokenizer.assert_called_once_with(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024 - max_tokens - 5)
        mock_inputs.to.assert_called_once_with(engine.model.device)
        engine.model.generate.assert_called_once()
        engine.tokenizer.decode.assert_called_once()

    async def test_generate_not_initialized(self, server_llm_engine_no_init_load):
        engine = server_llm_engine_no_init_load
        response = await engine.generate("Test prompt")
        assert response == "[LLM_ERROR: NOT_INITIALIZED]"

@pytest.mark.asyncio
class TestServerLLMEngineFineTune:

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_fine_tune_actor1_success(self, mock_to_thread, server_llm_engine_no_init_load, mock_db):
        engine = server_llm_engine_no_init_load
        engine.is_initialized = True
        # Ensure the model is a PeftModel mock that has train, eval, save_pretrained
        engine.model = MockGetPeftModel.return_value # From fixture, this is a fully mocked PeftModel
        engine.tokenizer = MagicMock(eos_token="<EOS>",
