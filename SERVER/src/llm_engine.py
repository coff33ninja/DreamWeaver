import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
import os
import asyncio # Added asyncio

from .config import ADAPTERS_PATH, MODELS_PATH # Ensure MODELS_PATH is available for base model caching

class LLMEngine:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", db=None): # Updated default model
        """
        Initialize the LLMEngine with a specified model and optional database handle.
        
        Sets up directory paths for LoRA adapters and base model caching, initializes model and tokenizer attributes, and attempts to load the model and tokenizer. The engine is configured for use with "Actor1" by default.
        """
        self.db = db
        self.model_name = model_name
        sane_model_name = self.model_name.replace("/", "_") # For path safety
        self.adapter_path = os.path.join(ADAPTERS_PATH, sane_model_name, "Actor1") # Server LLM is for Actor1
        self.base_model_cache_path = os.path.join(MODELS_PATH, "llm_base_models") # Centralized cache for base models

        os.makedirs(self.adapter_path, exist_ok=True)
        os.makedirs(self.base_model_cache_path, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.is_initialized = False # Flag to track initialization status
        self._load_model() # Initial load attempt

    def _load_model(self):
        """
        Load the tokenizer and model for text generation, applying quantization and LoRA adapter configuration as appropriate.
        
        If CUDA is available, loads the model with 4-bit quantization using BitsAndBytes and prepares it for k-bit training. Attempts to load existing LoRA adapters for "Actor1"; if none are found, initializes a new LoRA configuration targeting common LLaMA projection modules. Sets the model to evaluation mode and updates the initialization status. On failure, resets model state and initialization flag.
        """
        print(f"LLMEngine (Server/Actor1): Loading model '{self.model_name}'...")
        try:
            bnb_config = None
            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=False,
                )
                print("LLMEngine (Server/Actor1): CUDA available, using BitsAndBytes 4-bit quantization.")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.base_model_cache_path
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config if bnb_config else None, # Only if CUDA is available
                device_map="auto",
                trust_remote_code=True,
                cache_dir=self.base_model_cache_path
            )

            if torch.cuda.is_available(): # PEFT k-bit training prep only if CUDA
                self.model = prepare_model_for_kbit_training(self.model)

            # Try to load existing adapters for Actor1
            adapter_config_file = os.path.join(self.adapter_path, "adapter_config.json")
            if os.path.exists(adapter_config_file):
                print(f"LLMEngine (Server/Actor1): Loading existing LoRA adapters from {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            else:
                print(f"LLMEngine (Server/Actor1): No existing adapters found for Actor1 at {self.adapter_path}. Using base model or initializing new PEFT.")
                # Optionally initialize with a default LoRA config for future fine-tuning
                lora_config = LoraConfig(
                    r=16, lora_alpha=32,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Common for Llama-like models
                    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
                )
                self.model = get_peft_model(self.model, lora_config)

            self.model.eval() # Set to evaluation mode
            self.is_initialized = True
            print(f"LLMEngine (Server/Actor1): Model '{self.model_name}' loaded successfully. Device: {self.model.device}")
            if hasattr(self.model, 'print_trainable_parameters'):
                self.model.print_trainable_parameters()

        except Exception as e:
            print(f"LLMEngine (Server/Actor1): FATAL Error loading LLM model '{self.model_name}': {e}")
            self.model = None
            self.tokenizer = None
            self.is_initialized = False


    async def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Asynchronously generates a text completion for the given prompt using the loaded language model.
        
        Parameters:
            prompt (str): The input text prompt to generate a continuation for.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 150.
        
        Returns:
            str: The generated text continuation, or an error string if generation fails or the model is not initialized.
        """
        if not self.is_initialized or not self.model or not self.tokenizer:
            print("LLMEngine (Server/Actor1): Not initialized. Cannot generate text.")
            return "[LLM_ERROR: NOT_INITIALIZED]"

        def _blocking_generate():
            """
            Generates a text continuation for the given prompt using the loaded model and tokenizer.
            
            Returns:
                str: The generated text following the input prompt, or an error string if the model or tokenizer is not initialized.
            """
            if self.model is None or self.tokenizer is None:
                return "[LLM_ERROR: MODEL_OR_TOKENIZER_NOT_INITIALIZED]"
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.tokenizer.model_max_length - max_new_tokens - 5).to(self.model.device)
            with torch.no_grad():
                output_sequences = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, num_return_sequences=1,
                    do_sample=True, top_k=40, top_p=0.9, temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(output_sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        try:
            # print(f"LLMEngine (Server/Actor1): Generating response for prompt (first 50 chars): '{prompt[:50]}...'")
            response_text = await asyncio.to_thread(_blocking_generate)
            # print(f"LLMEngine (Server/Actor1): Generated response (first 50 chars): '{response_text[:50]}...'")
            return response_text
        except Exception as e:
            print(f"LLMEngine (Server/Actor1): Error during text generation: {e}")
            return "[LLM_ERROR: GENERATION_FAILED]"

    async def fine_tune(self, new_data_point: dict, Actor_id: str):
        """
        Asynchronously fine-tunes the model for Actor1 using new and existing training data.
        
        If a database is available, retrieves all training data for Actor1, appends the new data point if not already present, and performs LoRA-based fine-tuning in a background thread. Saves updated LoRA adapters after training. Only operates for Actor1; calls for other actors are ignored. The model is returned to evaluation mode after fine-tuning or on error.
        """
        if not self.is_initialized or not self.model or not self.tokenizer:
            print("LLMEngine (Server/Actor1): Not initialized. Cannot fine-tune.")
            return
        if Actor_id != "Actor1": # This engine instance is only for Actor1
            print(f"LLMEngine (Server/Actor1): Fine-tuning attempt for '{Actor_id}' ignored (engine is for Actor1).")
            return
        if not self.db:
            print("LLMEngine (Server/Actor1): Database not available. Cannot fetch full dataset for fine-tuning.")
            # Could proceed with just new_data_point if desired, but batch is better
            return

        print("LLMEngine (Server/Actor1): Initiating fine-tuning for Actor1...")

        # For this example, we'll use a simplified batch fine-tuning approach with all data for Actor1
        all_training_data = self.db.get_training_data_for_Actor("Actor1")
        if not all_training_data: # Should include the new_data_point if already saved
            print("LLMEngine (Server/Actor1): No training data found for Actor1. Adding current point.")
            all_training_data = [new_data_point] # Use the new point if no history
        elif new_data_point not in all_training_data : # Avoid duplicates if new_data_point is already in db
             all_training_data.append(new_data_point)


        if not all_training_data:
            print("LLMEngine (Server/Actor1): No training data to fine-tune with for Actor1.")
            return

        def _blocking_fine_tune():
            """
            Performs synchronous fine-tuning of the loaded model on all available training data for Actor1 using LoRA adapters.
            
            This method formats and tokenizes the training data, configures training parameters based on dataset size, runs the training loop, and saves the updated LoRA adapters if supported. The model is set to training mode during fine-tuning and returned to evaluation mode afterward.
            """
            if self.model is None or self.tokenizer is None:
                print("LLMEngine (Server/Actor1): Model or tokenizer not initialized. Cannot fine-tune.")
                return
            tokenizer = self.tokenizer  # Local reference for clarity
            self.model.train() # Set model to training mode

            formatted_texts = [f"{data['input']}{tokenizer.eos_token}{data['output']}{tokenizer.eos_token}" for data in all_training_data]
            dataset = Dataset.from_list([{"text": text} for text in formatted_texts])

            def tokenize_function(examples):
                """
                Tokenizes input text examples using the initialized tokenizer with truncation and padding.
                
                Parameters:
                    examples (dict): A dictionary containing a "text" key with input strings to tokenize.
                
                Returns:
                    dict: Tokenized representations of the input text.
                """
                if tokenizer is None:
                    raise RuntimeError("Tokenizer is not initialized.")
                return tokenizer(examples["text"], truncation=True, padding="max_length",
                                 max_length=tokenizer.model_max_length or 512) # Ensure max_length

            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

            # Adjust training arguments for potentially small datasets or single updates
            # These are example values and should be tuned.
            num_epochs = 1 if len(all_training_data) < 10 else 3
            batch_size = 1 if len(all_training_data) < 4 else 4
            ga_steps = max(1, 4 // batch_size)


            training_args = TrainingArguments(
                output_dir=os.path.join("./results_server_ft", Actor_id, self.model_name.replace("/", "_")),
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=ga_steps,
                num_train_epochs=num_epochs,
                learning_rate=5e-5, # Common for LoRA
                fp16=torch.cuda.is_available(), # True if CUDA, False otherwise
                logging_steps=max(1, len(tokenized_dataset) // (batch_size * ga_steps) // 2 +1), # Log a few times
                save_steps=max(1, len(tokenized_dataset) // (batch_size * ga_steps)), # Save once per effective epoch
                optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
                report_to="none",
                remove_unused_columns=False,
            )

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            trainer = Trainer(model=self.model, args=training_args, train_dataset=tokenized_dataset,
                              data_collator=data_collator)
            trainer.tokenizer = tokenizer  # Set tokenizer attribute directly for save_model/push_to_hub compatibility

            print(f"LLMEngine (Server/Actor1): Starting fine-tuning for Actor1 with {len(all_training_data)} data points...")
            trainer.train()

            # Save LoRA adapters
            os.makedirs(self.adapter_path, exist_ok=True)
            if hasattr(self.model, 'save_pretrained'): # Check if it's a PeftModel
                 self.model.save_pretrained(self.adapter_path)
                 print(f"LLMEngine (Server/Actor1): LoRA adapters for Actor1 saved to {self.adapter_path}")
            else:
                 print("LLMEngine (Server/Actor1): Model is not a PeftModel, cannot save adapters with save_pretrained.")

            self.model.eval() # Set model back to evaluation mode

        try:
            await asyncio.to_thread(_blocking_fine_tune)
            print("LLMEngine (Server/Actor1): Fine-tuning process completed for Actor1.")
        except Exception as e:
            print(f"LLMEngine (Server/Actor1): Error during fine-tuning for Actor1: {e}")
            if self.model:
                self.model.eval() # Ensure model is back in eval mode on error too

    # batch_fine_tune_Actor1 can be removed or refactored if fine_tune now handles batching from DB
    # For now, let's assume fine_tune is the primary async method.

if __name__ == "__main__":
    # Basic test (requires DB setup for fine-tuning part)
    async def main_test():
        """
        Asynchronously tests the LLMEngine's initialization and text generation using a dummy database.
        
        This function creates an instance of LLMEngine with a mock database, checks if the engine is initialized, and if so, generates and prints a response to a sample prompt. Fine-tuning is commented out and requires a real or mock database with data.
        """
        print("LLMEngine (Server/Actor1) Test:")
        # Dummy DB for testing generate, fine_tune would need a real one or mock
        class DummyDB:
            def get_training_data_for_Actor(self, Actor_id): """
Return an empty list as a placeholder for retrieving training data for the specified actor.

Parameters:
    Actor_id (str): The identifier of the actor for whom training data is requested.

Returns:
    list: An empty list, indicating no training data is available.
"""
return []

        engine = LLMEngine(db=DummyDB()) # Provide a dummy DB
        if engine.is_initialized:
            prompt = "Hello, what is your purpose?"
            print(f"Test Prompt: {prompt}")
            response = await engine.generate(prompt, max_new_tokens=50)
            print(f"Test Response: {response}")

            # Test fine_tune (will just log with DummyDB or try to add to empty list)
            # For a real test, you'd need a DB with some data.
            # print("\nTesting fine_tune (placeholder)...")
            # await engine.fine_tune({"input": "Test input", "output": "Test output"}, "Actor1")
        else:
            print("LLMEngine (Server/Actor1) failed to initialize for test.")

    asyncio.run(main_test())
