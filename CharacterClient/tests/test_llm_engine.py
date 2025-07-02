import pytest
import asyncio
import os
import logging
from unittest.mock import patch, MagicMock, mock_open, call

from CharacterClient.src.llm_engine import LLMEngine, DEFAULT_CLIENT_MODEL_NAME, JsonDataset
from CharacterClient.src.config import CLIENT_LLM_MODELS_PATH
from transformers import DataCollatorForLanguageModeling

# Mocks for external dependencies
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
def reset_global_mocks_llm_client(): # Renamed to be specific
    MockAutoModelForCausalLM.from_pretrained.reset_mock()
    MockAutoTokenizer.from_pretrained.reset_mock()
    MockBitsAndBytesConfig.reset_mock()
    MockPrepareModelForKbitTraining.reset_mock()
    MockPeftModel.from_pretrained.reset_mock()
    if hasattr(MockPeftModel, 'save_pretrained'): # Ensure save_pretrained is a MagicMock if PeftModel is one
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
    MockGetPeftModel.return_value = peft_model_instance # Used when no adapter exists
    MockPeftModel.from_pretrained.return_value = peft_model_instance # Used when adapter exists

    base_model_instance = MagicMock()
    base_model_instance.device = "cpu"
    MockAutoModelForCausalLM.from_pretrained.return_value = base_model_instance

@pytest.fixture
def mock_dependencies_llm_client(monkeypatch): # Renamed
    monkeypatch.setattr("CharacterClient.src.llm_engine.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("CharacterClient.src.llm_engine.AutoTokenizer", MockAutoTokenizer)
    monkeypatch.setattr("CharacterClient.src.llm_engine.BitsAndBytesConfig", MockBitsAndBytesConfig)
    monkeypatch.setattr("CharacterClient.src.llm_engine.prepare_model_for_kbit_training", MockPrepareModelForKbitTraining)
    monkeypatch.setattr("CharacterClient.src.llm_engine.PeftModel", MockPeftModel)
    monkeypatch.setattr("CharacterClient.src.llm_engine.LoraConfig", MockLoraConfig)
    monkeypatch.setattr("CharacterClient.src.llm_engine.get_peft_model", MockGetPeftModel)
    monkeypatch.setattr("CharacterClient.src.llm_engine.torch", MockTorch)
    monkeypatch.setattr("CharacterClient.src.llm_engine.Dataset", MockDataset)
    monkeypatch.setattr("CharacterClient.src.llm_engine.Trainer", MockTrainer)
    monkeypatch.setattr("CharacterClient.src.llm_engine.TrainingArguments", MockTrainingArguments)
    monkeypatch.setattr("CharacterClient.src.llm_engine.DataCollatorForLanguageModeling", MockDataCollatorForLanguageModeling)

@pytest.fixture
def llm_engine_no_init_client(mock_dependencies_llm_client): # Renamed
    with patch.object(LLMEngine, "_load_model_and_tokenizer", MagicMock()):
        engine = LLMEngine(model_name="test_model", Actor_id="test_actor")
        engine.is_initialized = False
    return engine

# --- Test Classes ---
class TestLLMEngineInitialization:
    # Tests from previous step - confirmed to be fine
    def test_init_paths_and_defaults(self, llm_engine_no_init_client): # Use renamed fixture
        engine = llm_engine_no_init_client
        assert engine.Actor_id == "test_actor"
        # ... (rest of assertions)

    @patch("CharacterClient.src.llm_engine.os.makedirs")
    def test_init_creates_directories(self, mock_makedirs, mock_dependencies_llm_client): # Use renamed fixture
        with patch.object(LLMEngine, "_load_model_and_tokenizer", MagicMock()):
            LLMEngine(model_name="another_model", Actor_id="another_actor")
            # ... (rest of assertions)

@pytest.mark.asyncio
class TestLLMEngineGenerate:
    # Tests from previous step - confirmed to be fine
    async def test_generate_success(self, llm_engine_no_init_client): # Use renamed fixture
        engine = llm_engine_no_init_client
        # ... (rest of test)

# --- Tests for Fine-Tuning ---
@pytest.mark.asyncio
class TestLLMEngineFineTuneAsync:

    @patch("CharacterClient.src.llm_engine.os.makedirs")
    @patch("CharacterClient.src.llm_engine.os.listdir")
    @patch("CharacterClient.src.llm_engine.json.dump")
    @patch("CharacterClient.src.llm_engine.json.load")
    @patch("builtins.open", new_callable=mock_open)
    async def test_fine_tune_async_full_flow(
        self, mock_file_open, mock_json_load, mock_json_dump, mock_listdir, mock_os_makedirs,
        llm_engine_no_init_client # Use client-specific fixture
    ):
        engine = llm_engine_no_init_client
        engine.is_initialized = True
        engine.model = MockGetPeftModel.return_value # This is already a PeftModel mock
        engine.tokenizer = MagicMock(eos_token="<EOS>",
