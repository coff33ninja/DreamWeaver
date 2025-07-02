import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import json

try:
    from peft import PeftModel, prepare_model_for_kbit_training
except ImportError as e:
    raise ImportError("The 'peft' package is required for LoRA/adapter support. Please install it with 'pip install peft'.") from e
import asyncio # Added asyncio
import logging

from .config import CLIENT_LLM_MODELS_PATH, ensure_client_directories

logger = logging.getLogger("dreamweaver_client")
ensure_client_directories() # This in config.py should also be logged if it prints
DEFAULT_CLIENT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        """
        Initialize the dataset with a list of data samples, a tokenizer, and a maximum token length.
        
        Parameters:
            data_list (list): List of training samples, each containing 'input' and 'output' fields.
            tokenizer: Tokenizer used to process text samples.
            max_length (int, optional): Maximum number of tokens per sample. Defaults to 512.
        """
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)
    def __getitem__(self, idx):
        """
        Tokenizes and returns the input and output fields of a data sample as model-ready tensors.
        
        The returned dictionary contains tokenized input tensors and a 'labels' tensor derived from the output text, suitable for supervised language model training.
        """
        item = self.data[idx]
        enc = self.tokenizer(
            item['input'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        label_enc = self.tokenizer(
            item['output'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc['labels'] = label_enc['input_ids'].squeeze(0)
        return enc

class LLMEngine:
    def __init__(self, model_name: str = "", Actor_id: str = "default_client_Actor"):
        """
        Initialize the LLMEngine with specified model and actor identifiers, setting up directory paths and loading the model and tokenizer.
        
        Parameters:
            model_name (str, optional): Name of the language model to load. Defaults to a preconfigured model if not provided.
            Actor_id (str, optional): Unique identifier for the client actor. Defaults to "default_client_Actor".
        """
        self.Actor_id = Actor_id or "default_client_Actor"
        self.model_name = model_name if model_name else DEFAULT_CLIENT_MODEL_NAME
        self.models_base_path = os.path.join(CLIENT_LLM_MODELS_PATH, "base_models")
        sane_model_name_for_path = self.model_name.replace("/", "_")
        self.adapters_path = os.path.join(CLIENT_LLM_MODELS_PATH, "adapters", self.Actor_id, sane_model_name_for_path)

        os.makedirs(self.models_base_path, exist_ok=True)
        os.makedirs(self.adapters_path, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        logger.info(f"Client LLMEngine: Initializing for model '{self.model_name}' for Actor_ID '{self.Actor_id}'. Base models path: {self.models_base_path}, Adapters path: {self.adapters_path}")
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Synchronously loads the tokenizer and model, applying quantization and LoRA adapter if available.
        
        Loads the tokenizer and causal language model from the specified model name and cache directory. If CUDA is available, applies 4-bit quantization and prepares the model for k-bit training. Attempts to load a LoRA adapter from disk if an adapter configuration is present. Sets the model to evaluation mode and updates the initialization status flag. Logs errors and updates the initialization flag if loading fails.
        """
        try:
            bnb_config = None
            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=False,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.models_base_path, trust_remote_code=True)
            if self.tokenizer and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config, # Will be None if CUDA not available
                device_map="auto",
                cache_dir=self.models_base_path,
                trust_remote_code=True
            )

            if torch.cuda.is_available() and self.model:
                self.model = prepare_model_for_kbit_training(self.model)

            adapter_config_file = os.path.join(self.adapters_path, "adapter_config.json")
            if os.path.exists(adapter_config_file) and self.model:
                try:
                    logger.info(f"Client LLMEngine ({self.Actor_id}): Found adapter config. Attempting to load LoRA adapters from {self.adapters_path}.")
                    self.model = PeftModel.from_pretrained(self.model, self.adapters_path)
                    logger.info(f"Client LLMEngine ({self.Actor_id}): Successfully loaded LoRA adapters from {self.adapters_path}.")
                except Exception as e:
                    logger.warning(f"Client LLMEngine ({self.Actor_id}): Error loading LoRA adapters from {self.adapters_path}: {e}. Proceeding with base model.", exc_info=True)

            if self.model:
                self.model.eval()
                self.is_initialized = True
                device_type = self.model.device.type if hasattr(self.model, 'device') else 'unknown'
                logger.info(f"Client LLMEngine for {self.Actor_id} (model: {self.model_name}) initialized successfully on device: {device_type}.")
            else:
                logger.error(f"Client LLMEngine ({self.Actor_id}): Model object is None after loading attempt for '{self.model_name}'.")
        except Exception as e:
            logger.critical(f"Client LLMEngine ({self.Actor_id}): FATAL Error during model loading for '{self.model_name}': {e}", exc_info=True)
            self.is_initialized = False

    async def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Asynchronously generates a text continuation for the given prompt using the loaded language model.
        
        Parameters:
            prompt (str): The input text prompt to generate a continuation for.
            max_new_tokens (int): The maximum number of new tokens to generate.
        
        Returns:
            str: The generated text continuation, or a special error string if the model is not initialized or generation fails.
        """
        if not self.is_initialized or not self.model or not self.tokenizer:
            logger.error(f"Client LLMEngine ({self.Actor_id}): Not initialized, cannot generate text for prompt: '{prompt[:50]}...'.")
            return "[LLM_ERROR:NOT_INITIALIZED]"

        logger.info(f"Client LLMEngine ({self.Actor_id}): Generating text for prompt: '{prompt[:100]}...'")

        def _blocking_generate_task():
            """
            Generates a text continuation for a given prompt using the loaded model and tokenizer.
            
            Returns:
                str: The generated text continuation, or "[LLM_NOT_INITIALIZED]" if the model or tokenizer is not loaded.
            """
            if not self.tokenizer or not self.model:
                return "[LLM_NOT_INITIALIZED]"
            max_prompt_len = (self.tokenizer.model_max_length or 2048) - max_new_tokens - 10
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(self.model.device)
            with torch.no_grad():
                output_sequences = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True, top_k=40, top_p=0.9, temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(output_sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        try:
            generated_text = await asyncio.to_thread(_blocking_generate_task)
            logger.info(f"Client LLMEngine ({self.Actor_id}): Generated text: '{generated_text[:100]}...'")
            return generated_text
        except Exception as e:
            logger.error(f"Client LLMEngine ({self.Actor_id}): Error during async text generation for prompt '{prompt[:50]}...': {e}", exc_info=True)
            return "[LLM_ERROR:GENERATION_FAILED]"

    async def fine_tune_async(self, training_data: dict, Actor_id_override: str = ""):
        """
        Asynchronously fine-tunes the loaded language model using provided training data and saves the updated model adapters.
        
        The method saves the given training sample locally, loads all available training samples for the current actor, and performs one epoch of supervised fine-tuning using HuggingFace Trainer. The fine-tuned model is saved to the actor's adapter directory. All blocking operations are executed in background threads to avoid blocking the event loop.
        
        Parameters:
            training_data (dict): A dictionary containing 'input' and 'output' fields for supervised fine-tuning.
            Actor_id_override (str, optional): If provided, overrides the default actor ID for data storage and adapter output.
        """
        current_Actor_id = Actor_id_override if Actor_id_override else self.Actor_id
        logger.info(f"Client LLMEngine ({current_Actor_id}): Async fine-tuning requested with data: {str(training_data)[:100]}...")

        def _save_data_blocking():
            """
            Save a training data sample as a JSON file in the local directory for the current actor.
            
            Creates the necessary directory if it does not exist and writes the provided training data to a uniquely named JSON file. Logs success or error messages.
            """
            data_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "training_data_local", current_Actor_id)
            os.makedirs(data_dir, exist_ok=True)
            timestamp = torch.randint(0, 1000000, (1,)).item()
            data_file = os.path.join(data_dir, f"training_sample_{timestamp}.json")
            try:
                with open(data_file, "w", encoding="utf-8") as f:
                    json.dump(training_data, f, ensure_ascii=False, indent=4)
                logger.info(f"Client LLMEngine ({current_Actor_id}): Saved training data sample to {data_file}")
            except Exception as e:
                logger.error(f"Client LLMEngine ({current_Actor_id}): Error saving local training data to {data_file}: {e}", exc_info=True)

        await asyncio.to_thread(_save_data_blocking)

        def _load_all_training_data():
            """
            Load all valid training data samples for the current actor from the local training data directory.
            
            Returns:
                all_data (list): A list of dictionaries, each containing 'input' and 'output' fields from valid JSON training samples.
            """
            data_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "training_data_local", current_Actor_id)
            all_data = []
            if not os.path.exists(data_dir):
                return all_data
            for fname in os.listdir(data_dir):
                if fname.endswith('.json'):
                    try:
                        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
                            sample = json.load(f)
                            if isinstance(sample, dict) and 'input' in sample and 'output' in sample:
                                all_data.append(sample)
                    except Exception as e:
                        logger.error(f"Client LLMEngine ({current_Actor_id}): Error loading training sample {fname}: {e}", exc_info=True)
            logger.info(f"Client LLMEngine ({current_Actor_id}): Loaded {len(all_data)} training samples from {data_dir}.")
            return all_data

        def _train_blocking():
            """
            Trains the current model on all available training data and saves the fine-tuned model to the adapter directory.
            
            If the model or tokenizer is not initialized, or if no training data is found, the function exits without performing training.
            """
            if not self.model or not self.tokenizer:
                logger.error(f"Client LLMEngine ({current_Actor_id}): Model or tokenizer not initialized, skipping fine-tuning.")
                return

            logger.info(f"Client LLMEngine ({current_Actor_id}): Loading all training data for fine-tuning...")
            all_data = _load_all_training_data()
            if not all_data:
                logger.info(f"Client LLMEngine ({current_Actor_id}): No training data found, skipping fine-tuning.")
                return

            dataset = JsonDataset(all_data, self.tokenizer)
            # Using self.adapters_path which is already defined as ".../CLIENT_LLM_MODELS_PATH/adapters/Actor_id/model_name/"
            # The "_finetuned" suffix might be redundant if this is the primary adapter path.
            # For consistency, let's use self.adapters_path directly for saving.
            output_dir = self.adapters_path
            # output_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "adapters", current_Actor_id, self.model_name.replace('/', '_') + "_finetuned")
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                save_steps=10,
                save_total_limit=2,
                logging_steps=5,
                learning_rate=2e-5,
                fp16=torch.cuda.is_available(),
                report_to=[],
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset
                # tokenizer=self.tokenizer  # Removed as Trainer does not accept this argument
            )
            logger.info(f"Client LLMEngine ({current_Actor_id}): Starting fine-tuning with {len(dataset)} samples. Output directory: {output_dir}")
            try:
                trainer.train()
                trainer.save_model(output_dir) # This saves the full model if not PeftModel, or adapters if PeftModel
                logger.info(f"Client LLMEngine ({current_Actor_id}): Fine-tuning complete. Model/adapters saved to {output_dir}")
            except Exception as e_train:
                 logger.error(f"Client LLMEngine ({current_Actor_id}): Error during model training or saving: {e_train}", exc_info=True)


        await asyncio.to_thread(_train_blocking)
        logger.info(f"Client LLMEngine ({current_Actor_id}): Async fine-tuning process complete.")

    def fine_tune(self, training_data: dict, Actor_id_override: str = ""):
        """
        Placeholder for synchronous fine-tuning; saves the provided training data as a JSON file locally.
        
        This method does not perform actual model fine-tuning. It only persists the training data for later use.
        
        Parameters:
            training_data (dict): The training sample to be saved.
            Actor_id_override (str, optional): If provided, overrides the default actor ID for data storage.
        """
        current_Actor_id = Actor_id_override if Actor_id_override else self.Actor_id
        logger.info(f"Client LLMEngine ({current_Actor_id}): Fine-tuning requested (SYNC placeholder). Data: {str(training_data)[:100]}...")
        data_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "training_data_local", current_Actor_id)
        os.makedirs(data_dir, exist_ok=True) # Handled by ensure_client_directories or earlier save, but good practice
        timestamp = torch.randint(0, 1000000, (1,)).item() # Still using torch for this random int
        data_file = os.path.join(data_dir, f"training_sample_sync_{timestamp}.json")
        try:
            # import json # Already imported at top level
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(training_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Client LLMEngine ({current_Actor_id}): Saved SYNC training data (placeholder) to {data_file}")
        except Exception as e:
            logger.error(f"Client LLMEngine ({current_Actor_id}): Error saving SYNC training data (placeholder) to {data_file}: {e}", exc_info=True)
        logger.warning("Client LLMEngine: Actual SYNC fine-tuning not implemented. Data saved only.")

    async def save_adapters_async(self):
        """
        Asynchronously saves LoRA adapters to disk if the current model is a PeftModel.
        
        This method runs the adapter saving operation in a separate thread to avoid blocking the event loop. If the model is not a PeftModel or is uninitialized, the adapters are not saved.
        """
        if self.model and isinstance(self.model, PeftModel):
            def _save_adapters_blocking():
                """
                Save the current model's adapters to the designated adapters path if the model is loaded.
                
                If the model is not initialized, logs a message indicating that adapters cannot be saved.
                """
                if self.model is not None:
                    logger.info(f"Client LLMEngine ({self.Actor_id}): Saving adapters to {self.adapters_path}...")
                    self.model.save_pretrained(self.adapters_path)
                    logger.info(f"Client LLMEngine ({self.Actor_id}): Adapters successfully saved to {self.adapters_path}.")
                else:
                    logger.error(f"Client LLMEngine ({self.Actor_id}): Cannot save adapters, model is None.")
            try:
                await asyncio.to_thread(_save_adapters_blocking)
            except Exception as e:
                logger.error(f"Client LLMEngine ({self.Actor_id}): Error saving LoRA adapters (async) to {self.adapters_path}: {e}", exc_info=True)
        else:
            logger.info(f"Client LLMEngine ({self.Actor_id}): Model is not a PeftModel or not initialized. Skipping adapter save (async).")

if __name__ == "__main__":
    # Setup basic logging for the test runner if this script is run directly
    if not logger.hasHandlers(): # Check if logger is already configured by main client setup
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')

    async def test_async_llm_engine():
        """
        Asynchronously tests the LLMEngine's text generation and fine-tuning capabilities.
        
        This function initializes an LLMEngine instance, generates a text response to a sample prompt asynchronously, and performs an asynchronous fine-tuning operation with test data. Results and progress are printed to the console.
        """
        logger.info("--- Client LLMEngine Async Test ---")
        engine = LLMEngine(Actor_id="test_Actor_async") # This will log its own init
        if engine.is_initialized:
            prompt = "Write a short poem about asynchronous programming."
            logger.info(f"\nTest Prompt: {prompt}")
            response = await engine.generate(prompt, max_new_tokens=60) # This will log
            logger.info(f"Async Response from LLM: {response}")

            logger.info("\nTesting async fine_tune (training placeholder)...")
            await engine.fine_tune_async({"input": "Async test input", "output": "Async test output"}) # This will log

            logger.info("\nTesting async save_adapters...")
            await engine.save_adapters_async() # This will log
        else:
            logger.error("Failed to initialize LLMEngine for async test.")
        logger.info("\n--- Client LLMEngine Async Test Complete ---")

    asyncio.run(test_async_llm_engine())
