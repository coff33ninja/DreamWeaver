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

from .config import CLIENT_LLM_MODELS_PATH, ensure_client_directories

ensure_client_directories()
DEFAULT_CLIENT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        """
        Initialize the dataset with a list of input-output samples, a tokenizer, and a maximum token length.
        
        Parameters:
            data_list (list): List of dictionaries, each containing 'input' and 'output' text samples.
            tokenizer: Tokenizer used to process the text samples.
            max_length (int, optional): Maximum number of tokens for input and output sequences. Defaults to 512.
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
        Tokenizes and returns the input and output texts for a given index as model-ready tensors.
        
        The returned dictionary contains tokenized input tensors and a 'labels' tensor for supervised training.
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
        Initialize the LLMEngine with specified model and actor identifiers, setting up model and adapter directories and loading the model and tokenizer.
        
        Parameters:
            model_name (str, optional): Name of the pretrained model to load. Defaults to a predefined model if not specified.
            Actor_id (str, optional): Identifier for the actor/client instance. Used to organize adapter storage. Defaults to "default_client_Actor".
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
        print(f"Client LLMEngine: Initializing for model '{self.model_name}' for Actor_ID '{self.Actor_id}'")
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Synchronously loads the tokenizer and language model, applying quantization and LoRA adapters if available.
        
        Initializes the tokenizer and model from pretrained weights, sets up 4-bit quantization if CUDA is available, and prepares the model for k-bit training. Attempts to load LoRA adapters from the specified adapter path if present. Sets the model to evaluation mode and updates the initialization status.
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
                    self.model = PeftModel.from_pretrained(self.model, self.adapters_path)
                except Exception as e:
                    print(f"Client LLMEngine: Error loading LoRA adapters from {self.adapters_path}: {e}. Proceeding with base model.")

            if self.model:
                self.model.eval()
                self.is_initialized = True
                print(f"Client LLMEngine for {self.Actor_id} initialized successfully.")
            else:
                print(f"Client LLMEngine: Model failed to load for '{self.model_name}'")
        except Exception as e:
            print(f"Client LLMEngine: FATAL Error during model loading for '{self.model_name}': {e}")
            self.is_initialized = False

    async def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Asynchronously generates text from a given prompt using the loaded language model.
        
        Parameters:
            prompt (str): The input text prompt to generate a continuation for.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 150).
        
        Returns:
            str: The generated text continuation, or an error string if the model is not initialized or generation fails.
        """
        if not self.is_initialized or not self.model or not self.tokenizer:
            print("Client LLMEngine: Not initialized, cannot generate text.")
            return "[LLM_NOT_INITIALIZED]"

        def _blocking_generate_task():
            """
            Generates a text continuation from a prompt using the loaded model and tokenizer in a blocking manner.
            
            Returns:
                str: The generated text, or "[LLM_NOT_INITIALIZED]" if the model or tokenizer is not initialized.
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
            return generated_text
        except Exception as e:
            print(f"Client LLMEngine ({self.Actor_id}): Error during async text generation: {e}")
            return "[LLM_GENERATION_ERROR]"

    async def fine_tune_async(self, training_data: dict, Actor_id_override: str = ""):
        """
        Asynchronously fine-tunes the loaded language model using locally stored training samples and saves the resulting adapters.
        
        The provided training data sample is saved to a local directory, all available samples are loaded, and the HuggingFace Trainer is used to perform one epoch of fine-tuning. The fine-tuned model is saved as a LoRA adapter for the current or overridden actor ID.
        """
        current_Actor_id = Actor_id_override if Actor_id_override else self.Actor_id
        print(f"Client LLMEngine ({current_Actor_id}): Async fine-tuning requested.")

        def _save_data_blocking():
            """
            Save a training data sample as a JSON file in the local training data directory for the current actor.
            
            Creates the directory if it does not exist and writes the provided training data to a uniquely named file.
            """
            data_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "training_data_local", current_Actor_id)
            os.makedirs(data_dir, exist_ok=True)
            timestamp = torch.randint(0, 1000000, (1,)).item()
            data_file = os.path.join(data_dir, f"training_sample_{timestamp}.json")
            try:
                with open(data_file, "w", encoding="utf-8") as f:
                    json.dump(training_data, f, ensure_ascii=False, indent=4)
                print(f"Client LLMEngine ({current_Actor_id}): Saved training data sample to {data_file}")
            except Exception as e:
                print(f"Client LLMEngine ({current_Actor_id}): Error saving local training data: {e}")

        await asyncio.to_thread(_save_data_blocking)

        def _load_all_training_data():
            """
            Load all valid training data samples for the current actor from the local training data directory.
            
            Returns:
                all_data (list): A list of dictionaries, each containing 'input' and 'output' keys from JSON files.
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
                        print(f"Error loading training sample {fname}: {e}")
            return all_data

        def _train_blocking():
            """
            Trains the current model on all available local training data samples and saves the fine-tuned model to the adapter directory.
            
            Skips training if the model or tokenizer is not initialized, or if no training data is found.
            """
            if not self.model or not self.tokenizer:
                print(f"Client LLMEngine ({current_Actor_id}): Model or tokenizer not initialized, skipping training.")
                return
            all_data = _load_all_training_data()
            if not all_data:
                print(f"Client LLMEngine ({current_Actor_id}): No training data found, skipping training.")
                return
            dataset = JsonDataset(all_data, self.tokenizer)
            output_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "adapters", current_Actor_id, self.model_name.replace('/', '_') + "_finetuned")
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
            print(f"Client LLMEngine ({current_Actor_id}): Starting training with {len(dataset)} samples...")
            trainer.train()
            trainer.save_model(output_dir)
            print(f"Client LLMEngine ({current_Actor_id}): Training complete. Model saved to {output_dir}")

        await asyncio.to_thread(_train_blocking)
        print(f"Client LLMEngine ({current_Actor_id}): Async fine-tuning complete.")

    def fine_tune(self, training_data: dict, Actor_id_override: str = ""):
        """
        Placeholder method for synchronous fine-tuning; saves the provided training data to a local JSON file but does not perform actual model training.
        
        Parameters:
            training_data (dict): A dictionary containing input and output fields for training.
            Actor_id_override (str, optional): If provided, overrides the default actor ID for data storage.
        """
        current_Actor_id = Actor_id_override if Actor_id_override else self.Actor_id
        print(f"Client LLMEngine ({current_Actor_id}): Fine-tuning requested (SYNC placeholder).")
        data_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "training_data_local", current_Actor_id)
        os.makedirs(data_dir, exist_ok=True)
        timestamp = torch.randint(0, 1000000, (1,)).item()
        data_file = os.path.join(data_dir, f"training_sample_sync_{timestamp}.json")
        try:
            import json
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(training_data, f, ensure_ascii=False, indent=4)
            print(f"Client LLMEngine ({current_Actor_id}): Saved SYNC training data to {data_file}")
        except Exception as e:
            print(f"Client LLMEngine ({current_Actor_id}): Error saving SYNC training data: {e}")
        print("Client LLMEngine: Actual SYNC fine-tuning not implemented.")

    async def save_adapters_async(self):
        """
        Asynchronously saves LoRA adapters to the configured adapter path if the current model is a PeftModel.
        """
        if self.model and isinstance(self.model, PeftModel):
            def _save_adapters_blocking():
                """
                Save the current model's adapters to the designated adapters path if the model is initialized.
                
                If the model is not initialized, logs a message indicating that adapters cannot be saved.
                """
                if self.model is not None:
                    self.model.save_pretrained(self.adapters_path)
                else:
                    print(f"Client LLMEngine ({self.Actor_id}): Cannot save adapters, model is None.")
            try:
                await asyncio.to_thread(_save_adapters_blocking)
                print(f"Client LLMEngine ({self.Actor_id}): LoRA adapters saved (async) to {self.adapters_path}")
            except Exception as e:
                print(f"Client LLMEngine ({self.Actor_id}): Error saving LoRA adapters (async): {e}")
        else:
            print(f"Client LLMEngine ({self.Actor_id}): Model not PeftModel or not init, cannot save adapters (async).")

if __name__ == "__main__":
    async def test_async_llm_engine():
        """
        Demonstrates asynchronous usage of the LLMEngine, including text generation and fine-tuning.
        
        Runs a test routine that initializes the LLMEngine, generates a response to a sample prompt asynchronously, and performs a placeholder asynchronous fine-tuning operation.
        """
        print("--- Client LLMEngine Async Test ---")
        engine = LLMEngine(Actor_id="test_Actor_async")
        if engine.is_initialized:
            prompt = "Write a short poem about asynchronous programming."
            print(f"\nTest Prompt: {prompt}")
            response = await engine.generate(prompt, max_new_tokens=60)
            print(f"Async Response: {response}")

            print("\nTesting async fine_tune (placeholder)...")
            await engine.fine_tune_async({"input": "Async test input", "output": "Async test output"})
            # await engine.save_adapters_async() # Test saving (would only work if adapters were loaded/created)
        else:
            print("Failed to initialize LLMEngine for async test.")
        print("\n--- Client LLMEngine Async Test Complete ---")

    asyncio.run(test_async_llm_engine())
