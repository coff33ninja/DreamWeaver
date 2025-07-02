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
MockGetPeftModel = MagicMock() # Used if adapter doesn't exist
MockDataset = MagicMock()
MockTrainer = MagicMock()
MockTrainingArguments = MagicMock()
MockTorch = MagicMock()
MockDataCollator = MagicMock()


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
    monkeypatch.setattr("SERVER.src.llm_engine.DataCollatorForLanguageModeling", MockDataCollator)


    MockAutoModelForCausalLM.from_pretrained.reset_mock()
    MockAutoTokenizer.from_pretrained.reset_mock()
    MockBitsAndBytesConfig.reset_mock()
    MockPrepareModelForKbitTraining.reset_mock()
    MockPeftModel.from_pretrained.reset_mock()
    MockGetPeftModel.reset_mock()
    MockLoraConfig.reset_mock()
    MockTorch.cuda.is_available.return_value = False
    MockTorch.cuda.is_bf16_supported.return_value = False
    MockGetPeftModel.return_value = MagicMock() # Ensure get_peft_model returns a mock model


@pytest.fixture
def mock_db():
    """Fixture for a mocked database object."""
    db = MagicMock()
    db.get_training_data_for_Actor.return_value = [] # Default to no training data
    return db

@pytest.fixture
async def server_llm_engine_no_init_load(mock_server_dependencies, mock_db, event_loop):
    """
    Provides a server LLMEngine instance where _load_model is NOT called automatically.
    asyncio.get_event_loop().run_in_executor is patched.
    """
    with patch.object(LLMEngine, "_load_model", MagicMock()) as mock_load_method_on_instance:
    # with patch("asyncio.get_event_loop") as mock_get_loop:
        # mock_event_loop = MagicMock()
        # mock_event_loop.run_in_executor = MagicMock() # Prevent _load_model from running
        # mock_get_loop.return_value = mock_event_loop
        engine = LLMEngine(model_name="server_test_model", db=mock_db)
        engine.is_initialized = False # Explicitly set for clarity
    return engine


class TestServerLLMEngineInitialization:

    def test_init_paths_and_defaults(self, mock_server_dependencies, server_llm_engine_no_init_load, mock_db):
        engine = server_llm_engine_no_init_load # _load_model is mocked out

        assert engine.db == mock_db
        assert engine.model_name == "server_test_model"
        sane_model_name = "server_test_model".replace("/", "_")
        expected_adapter_path = os.path.join(ADAPTERS_PATH, sane_model_name, "Actor1")
        assert engine.adapter_path == expected_adapter_path
        assert engine.base_model_cache_path == os.path.join(MODELS_PATH, "llm_base_models")
        assert not engine.is_initialized
        # Check that _load_model was patched out correctly by the fixture for this instance
        assert isinstance(engine._load_model, MagicMock)


    @patch("SERVER.src.llm_engine.os.makedirs")
    def test_init_creates_directories(self, mock_makedirs, mock_server_dependencies, mock_db):
        # Instantiate LLMEngine directly to test its __init__'s os.makedirs calls
        # Patch run_in_executor to prevent _load_model from actually running and interfering
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


    def test_load_model_success_no_cuda_no_adapter_creates_peft(self, mock_server_dependencies, server_llm_engine_no_init_load):
        engine = server_llm_engine_no_init_load
        # Reset the _load_model mock to be a real method for this test, but keep dependencies mocked
        engine._load_model = LLMEngine._load_model.__get__(engine, LLMEngine) # Unpatch for this instance

        MockTorch.cuda.is_available.return_value = False
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "[EOS]"
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        MockAutoModelForCausalLM.from_pretrained.return_value = mock_model_instance
        
        mock_peft_model_after_init = MockGetPeftModel.return_value # what get_peft_model returns
        
        with patch("SERVER.src.llm_engine.os.path.exists", return_value=False): # No adapter config
            engine._load_model()

        MockAutoTokenizer.from_pretrained.assert_called_once_with(engine.model_name, trust_remote_code=True, cache_dir=engine.base_model_cache_path)
        assert mock_tokenizer_instance.pad_token == "[EOS]"
        MockAutoModelForCausalLM.from_pretrained.assert_called_once_with(
            engine.model_name,
            quantization_config=None,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=engine.base_model_cache_path
        )
        MockPrepareModelForKbitTraining.assert_not_called()
        MockPeftModel.from_pretrained.assert_not_called() # No adapter exists
        MockLoraConfig.assert_called_once() # Should init new LoraConfig
        MockGetPeftModel.assert_called_once_with(mock_model_instance, MockLoraConfig.return_value)
        
        mock_peft_model_after_init.eval.assert_called_once()
        assert engine.is_initialized
        assert engine.model == mock_peft_model_after_init # Model should be the one from get_peft_model
        assert engine.tokenizer == mock_tokenizer_instance


    def test_load_model_success_with_cuda_and_existing_adapter(self, mock_server_dependencies, server_llm_engine_no_init_load):
        engine = server_llm_engine_no_init_load
        engine._load_model = LLMEngine._load_model.__get__(engine, LLMEngine) # Unpatch

        MockTorch.cuda.is_available.return_value = True
        MockTorch.cuda.is_bf16_supported.return_value = True

        mock_tokenizer_instance = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_base_model_instance = MagicMock()
        MockAutoModelForCausalLM.from_pretrained.return_value = mock_base_model_instance
        
        mock_loaded_peft_model_instance = MagicMock() # This is what PeftModel.from_pretrained returns
        MockPeftModel.from_pretrained.return_value = mock_loaded_peft_model_instance
        
        MockPrepareModelForKbitTraining.return_value = mock_base_model_instance # Simulate return

        with patch("SERVER.src.llm_engine.os.path.exists", return_value=True): # Adapter config exists
            engine._load_model()

        MockBitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=MockTorch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        MockAutoModelForCausalLM.from_pretrained.assert_called_once() # Args checked by bnb_config mock
        MockPrepareModelForKbitTraining.assert_called_once_with(mock_base_model_instance)
        MockPeftModel.from_pretrained.assert_called_once_with(mock_base_model_instance, engine.adapter_path)
        MockGetPeftModel.assert_not_called() # Should not init new Lora if adapter loaded

        mock_loaded_peft_model_instance.eval.assert_called_once()
        assert engine.is_initialized
        assert engine.model == mock_loaded_peft_model_instance


@pytest.mark.asyncio
class TestServerLLMEngineGenerate:
    async def test_generate_success(self, mock_server_dependencies, server_llm_engine_no_init_load):
        engine = server_llm_engine_no_init_load
        engine.is_initialized = True # Manually set as _load_model is mocked by fixture
        engine.model = MagicMock()
        engine.tokenizer = MagicMock()

        mock_inputs = {"input_ids": MagicMock(shape=(-1, 5))}
        engine.tokenizer.return_value = mock_inputs
        engine.tokenizer.model_max_length = 1024 # Example value
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
        engine.model.generate.assert_called_once()
        engine.tokenizer.decode.assert_called_once()

    async def test_generate_not_initialized(self, server_llm_engine_no_init_load):
        engine = server_llm_engine_no_init_load # is_initialized = False
        response = await engine.generate("Test prompt")
        assert response == "[LLM_ERROR: NOT_INITIALIZED]"


@pytest.mark.asyncio
class TestServerLLMEngineFineTune:
    async def test_fine_tune_actor1_success(self, mock_server_dependencies, server_llm_engine_no_init_load, mock_db):
        engine = server_llm_engine_no_init_load
        engine.is_initialized = True # Manually set
        engine.model = MockPeftModel() # Needs to be PeftModel for save_pretrained
        engine.tokenizer = MagicMock()
        engine.tokenizer.eos_token = "<EOS>"
