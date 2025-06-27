import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Assuming client's config.py is in the same directory or accessible via PYTHONPATH
from .config import CLIENT_LLM_MODELS_PATH, ensure_client_directories

# Ensure directories are created when this module is loaded
ensure_client_directories()

# Default model if not specified by character traits
DEFAULT_CLIENT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # A small, capable model

class LLMEngine:
    def __init__(self, model_name: str = None, pc_id: str = "default_client_pc"):
        self.pc_id = pc_id # For pc-specific adapters if fine-tuning is implemented
        self.model_name = model_name if model_name else DEFAULT_CLIENT_MODEL_NAME

        # Base path for all LLM models on the client
        self.models_base_path = os.path.join(CLIENT_LLM_MODELS_PATH, "base_models")
        # Path for LoRA adapters, specific to this pc_id and model
        # Sanitize model name for path by replacing slashes, common in HF model IDs
        sane_model_name_for_path = self.model_name.replace("/", "_")
        self.adapters_path = os.path.join(CLIENT_LLM_MODELS_PATH, "adapters", self.pc_id, sane_model_name_for_path)

        os.makedirs(self.models_base_path, exist_ok=True)
        os.makedirs(self.adapters_path, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.is_initialized = False

        print(f"Client LLMEngine: Initializing for model '{self.model_name}' for PC_ID '{self.pc_id}'")
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
                bnb_4bit_use_double_quant=False,
            )

            # Transformers library will handle caching to this directory.
            # If model files are already here from a previous download, it will use them.
            print(f"Client LLMEngine: Attempting to load/download base model '{self.model_name}' using cache_dir '{self.models_base_path}'.")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.models_base_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Client LLMEngine: Tokenizer pad_token set to eos_token.")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config if torch.cuda.is_available() else None, # Only use BNB on CUDA
                device_map="auto",
                cache_dir=self.models_base_path,
                trust_remote_code=True
            )

            print(f"Client LLMEngine: Base model '{self.model_name}' loaded. Device: {self.model.device}")

            if torch.cuda.is_available():
                 self.model = prepare_model_for_kbit_training(self.model)

            # Check if adapters exist and load them
            adapter_config_file = os.path.join(self.adapters_path, "adapter_config.json")
            if os.path.exists(adapter_config_file):
                try:
                    print(f"Client LLMEngine: Found existing LoRA adapters at {self.adapters_path}. Loading them.")
                    self.model = PeftModel.from_pretrained(self.model, self.adapters_path)
                    print(f"Client LLMEngine: LoRA adapters loaded successfully.")
                except Exception as e:
                    print(f"Client LLMEngine: Error loading LoRA adapters from {self.adapters_path}: {e}. Proceeding with base model.")
            else:
                print(f"Client LLMEngine: No LoRA adapters found at {self.adapters_path}. Using base model.")
                # To enable fine-tuning later, you might want to apply a default LoraConfig here
                # if no adapters were found, so the model becomes a PeftModel.
                # Example:
                # print("Client LLMEngine: Initializing with default LoRA config for future fine-tuning.")
                # lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
                # self.model = get_peft_model(self.model, lora_config)

            self.model.eval()
            self.is_initialized = True
            print(f"Client LLMEngine for {self.pc_id} initialized. Trainable params:")
            if hasattr(self.model, 'print_trainable_parameters'):
                self.model.print_trainable_parameters()

        except Exception as e:
            print(f"Client LLMEngine: FATAL Error during model loading for '{self.model_name}': {e}")
            self.model = None
            self.tokenizer = None
            self.is_initialized = False

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str: # Increased default
        if not self.is_initialized or not self.model or not self.tokenizer:
            print("Client LLMEngine: Not initialized, cannot generate text.")
            return "[LLM_NOT_INITIALIZED]"

        try:
            # Ensure prompt is not excessively long, respecting tokenizer's model_max_length
            # Max length for tokenization should consider current prompt length and max_new_tokens
            max_prompt_len = self.tokenizer.model_max_length - max_new_tokens - 10 # -10 for safety margin
            if len(self.tokenizer.encode(prompt)) > max_prompt_len :
                 print(f"Client LLMEngine: Warning - Prompt is too long. Truncating to {max_prompt_len} tokens.")


            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(self.model.device)

            with torch.no_grad():
                output_sequences = self.model.generate(
                    **inputs, # Pass all inputs from tokenizer
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=40, # Adjusted
                    top_p=0.9, # Adjusted
                    temperature=0.8, # Adjusted
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(output_sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return generated_text.strip()
        except Exception as e:
            print(f"Client LLMEngine: Error during text generation: {e}")
            return "[LLM_GENERATION_ERROR]"

    def fine_tune(self, training_data: dict, pc_id_override: str = None):
        current_pc_id = pc_id_override if pc_id_override else self.pc_id
        print(f"Client LLMEngine ({current_pc_id}): Fine-tuning requested (currently placeholder).")

        data_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "training_data_local", current_pc_id)
        os.makedirs(data_dir, exist_ok=True)
        # Create a somewhat unique filename
        timestamp = torch.randint(0, 1000000, (1,)).item() # For a simple unique component
        data_file = os.path.join(data_dir, f"training_sample_{timestamp}.json")
        try:
            import json
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(training_data, f, ensure_ascii=False, indent=4)
            print(f"Client LLMEngine ({current_pc_id}): Saved training data sample to {data_file}")
        except Exception as e:
            print(f"Client LLMEngine ({current_pc_id}): Error saving local training data: {e}")

        # TODO: Implement actual fine-tuning logic here if desired for client-side evolution
        # This would involve:
        # 1. Checking if self.model is a PeftModel. If not, apply a LoraConfig.
        #    if not isinstance(self.model, PeftModel):
        #        lora_config = LoraConfig(...) # Define your LoraConfig
        #        self.model = get_peft_model(self.model, lora_config)
        #        self.model.print_trainable_parameters()
        # 2. Set model to train mode: self.model.train()
        # 3. Accumulating data, preparing Dataset, DataLoader.
        # 4. Optimizer setup.
        # 5. Training loop.
        # 6. Saving adapters: self.save_adapters()
        # 7. Set model back to eval mode: self.model.eval()
        print("Client LLMEngine: Actual fine-tuning process is not yet implemented on client-side.")
        pass


    def save_adapters(self):
        """Saves LoRA adapters if the model is a PeftModel."""
        if self.model and isinstance(self.model, PeftModel): # Check if it's a PeftModel
            try:
                self.model.save_pretrained(self.adapters_path)
                print(f"Client LLMEngine ({self.pc_id}): LoRA adapters saved to {self.adapters_path}")
            except Exception as e:
                print(f"Client LLMEngine ({self.pc_id}): Error saving LoRA adapters: {e}")
        else:
            print(f"Client LLMEngine ({self.pc_id}): Model is not a PeftModel or not initialized, cannot save adapters. Current type: {type(self.model)}")


if __name__ == "__main__":
    print("--- Client LLMEngine Test ---")

    print(f"\nTesting with default model: {DEFAULT_CLIENT_MODEL_NAME}")
    client_llm_default = LLMEngine(pc_id="test_pc_main") # Use a distinct pc_id for testing
    if client_llm_default.is_initialized:
        prompts = [
            "Hello, can you tell me a short story?",
            "What is your favorite color and why?",
            "Translate 'good morning' into French."
        ]
        for i, p in enumerate(prompts):
            print(f"\nTest {i+1} - Prompt: {p}")
            response = client_llm_default.generate(p, max_new_tokens=70)
            print(f"Response: {response}")

        print("\nTesting placeholder fine-tuning...")
        client_llm_default.fine_tune({"input": "User: Hi there!", "output": "Bot: Hello! How can I help you today?"})

        # Test saving adapters (will only work if it became a PeftModel, e.g. by loading existing or initing new)
        # client_llm_default.save_adapters()
    else:
        print(f"Failed to initialize LLMEngine with default model for test_pc_main.")

    print("\n--- Client LLMEngine Test Complete ---")
