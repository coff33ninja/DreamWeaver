import pytest
import asyncio
import os
import logging
from unittest.mock import patch, MagicMock, mock_open, AsyncMock, call

# Ensure correct import path for LLMEngine from SERVER/src/
from SERVER.src.llm_engine import LLMEngine
from SERVER.src.config import ADAPTERS_PATH, MODELS_PATH

# Mock PEFT and Transformers objects
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

@pytest.fixture(autouse=True)
def reset_global_mocks_llm_server(): # Renamed for server
    MockAutoModelForCausalLM.from_pretrained.reset_mock()
    MockAutoTokenizer.from_pretrained.reset_mock()
    MockBitsAndBytesConfig.reset_mock()
    MockPrepareModelForKbitTraining.reset_mock()
    MockPeftModel.from_pretrained.reset_mock()
    if hasattr(MockPeftModel, 'save_pretrained'):
        MockPeftModel.save_pretrained.reset_mock()
    MockGetPeftModel.reset_mock()
    MockLoraConfig.reset_mock()
    MockDataCollatorForLanguageModeling.reset_mock()
    MockTrainer.reset_mock()
    MockTrainingArguments.reset_mock()
    MockDataset.from_list.reset_mock()
    MockTorch.cuda.is_available.return_value = False
    MockTorch.cuda.is_bf16_supported.return_value = False

    peft_model_instance = MagicMock(spec=MockPeftModel)
    peft_model_instance.save_pretrained = MagicMock()
    peft_model_instance.eval = MagicMock()
    peft_model_instance.train = MagicMock()
    peft_model_instance.device = "cpu"
    MockGetPeftModel.return_value = peft_model_instance
    MockPeftModel.from_pretrained.return_value = peft_model_instance # If adapter is loaded

    base_model_instance = MagicMock()
    base_model_instance.device = "cpu"
    MockAutoModelForCausalLM.from_pretrained.return_value = base_model_instance

@pytest.fixture
def mock_server_dependencies_llm(monkeypatch): # Renamed
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
def mock_db_llm(): # Renamed
    db = MagicMock()
    db.get_training_data_for_Actor.return_value = []
    return db

@pytest.fixture
def server_llm_engine_no_init(mock_server_dependencies_llm, mock_db_llm): # Renamed
    with patch.object(LLMEngine, "_load_model", MagicMock()):
        engine = LLMEngine(model_name="server_test_model", db=mock_db_llm)
        engine.is_initialized = False
    return engine

# --- Test Classes (Initialization and Generate are similar to client, adapt if needed) ---
class TestServerLLMEngineInitialization:
    # ... (Assuming these tests are similar to client and were correct)
    def test_init_paths_and_defaults(self, server_llm_engine_no_init, mock_db_llm):
        engine = server_llm_engine_no_init
        assert engine.db == mock_db_llm
        # ... rest of assertions

@pytest.mark.asyncio
class TestServerLLMEngineGenerate:
    # ... (Assuming these tests are similar to client and were correct)
    async def test_generate_success(self, server_llm_engine_no_init):
        engine = server_llm_engine_no_init
        # ... rest of test

@pytest.mark.asyncio
class TestServerLLMEngineFineTune:

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_fine_tune_actor1_success(self, mock_to_thread, server_llm_engine_no_init, mock_db_llm):
        engine = server_llm_engine_no_init
        engine.is_initialized = True
        engine.model = MockGetPeftModel.return_value # This is a PeftModel mock
        engine.tokenizer = MagicMock(eos_token="<EOS>",
