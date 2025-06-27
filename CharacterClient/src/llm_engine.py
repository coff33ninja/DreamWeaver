import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import asyncio # Added asyncio

from .config import CLIENT_LLM_MODELS_PATH, ensure_client_directories

ensure_client_directories()
DEFAULT_CLIENT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class LLMEngine:
    def __init__(self, model_name: str = None, pc_id: str = "default_client_pc"):
        self.pc_id = pc_id
        self.model_name = model_name if model_name else DEFAULT_CLIENT_MODEL_NAME
        self.models_base_path = os.path.join(CLIENT_LLM_MODELS_PATH, "base_models")
        sane_model_name_for_path = self.model_name.replace("/", "_")
        self.adapters_path = os.path.join(CLIENT_LLM_MODELS_PATH, "adapters", self.pc_id, sane_model_name_for_path)

        os.makedirs(self.models_base_path, exist_ok=True)
        os.makedirs(self.adapters_path, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        # _load_model_and_tokenizer is blocking, called during init.
        # For a truly non-blocking init, this would need to be made async and awaited,
        # or run in a separate thread with a future to signal completion.
        # For now, keeping init blocking as per common library patterns.
        print(f"Client LLMEngine: Initializing for model '{self.model_name}' for PC_ID '{self.pc_id}'")
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Synchronous/blocking model loading method."""
        try:
            bnb_config = None
            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=False,
                )

            # print(f"Client LLMEngine: Attempting to load/download base model '{self.model_name}' using cache_dir '{self.models_base_path}'.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.models_base_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config, # Will be None if CUDA not available
                device_map="auto",
                cache_dir=self.models_base_path,
                trust_remote_code=True
            )
            # print(f"Client LLMEngine: Base model '{self.model_name}' loaded. Device: {self.model.device}")

            if torch.cuda.is_available():
                 self.model = prepare_model_for_kbit_training(self.model)

            adapter_config_file = os.path.join(self.adapters_path, "adapter_config.json")
            if os.path.exists(adapter_config_file):
                try:
                    # print(f"Client LLMEngine: Found existing LoRA adapters at {self.adapters_path}. Loading them.")
                    self.model = PeftModel.from_pretrained(self.model, self.adapters_path)
                    # print(f"Client LLMEngine: LoRA adapters loaded successfully.")
                except Exception as e:
                    print(f"Client LLMEngine: Error loading LoRA adapters from {self.adapters_path}: {e}. Proceeding with base model.")
            # else:
                # print(f"Client LLMEngine: No LoRA adapters found at {self.adapters_path}. Using base model.")

            self.model.eval()
            self.is_initialized = True
            print(f"Client LLMEngine for {self.pc_id} initialized successfully.")
            # if hasattr(self.model, 'print_trainable_parameters'):
            #     self.model.print_trainable_parameters()

        except Exception as e:
            print(f"Client LLMEngine: FATAL Error during model loading for '{self.model_name}': {e}")
            self.is_initialized = False


    async def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        if not self.is_initialized or not self.model or not self.tokenizer:
            print("Client LLMEngine: Not initialized, cannot generate text.")
            return "[LLM_NOT_INITIALIZED]"

        def _blocking_generate_task():
            # Ensure prompt is not excessively long
            max_prompt_len = (self.tokenizer.model_max_length or 2048) - max_new_tokens - 10

            # Truncate prompt if too long - this is a basic way, more sophisticated needed for context
            # For now, just a simple warning and direct tokenization.
            # encoded_prompt = self.tokenizer.encode(prompt)
            # if len(encoded_prompt) > max_prompt_len:
            #     print(f"Client LLMEngine: Warning - Prompt length ({len(encoded_prompt)}) exceeds max_prompt_len ({max_prompt_len}). Truncation might occur by tokenizer.")

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
            # print(f"Client LLMEngine ({self.pc_id}): Generating response async for prompt: '{prompt[:30]}...'")
            generated_text = await asyncio.to_thread(_blocking_generate_task)
            # print(f"Client LLMEngine ({self.pc_id}): Generated text: '{generated_text[:30]}...'")
            return generated_text
        except Exception as e:
            print(f"Client LLMEngine ({self.pc_id}): Error during async text generation: {e}")
            return "[LLM_GENERATION_ERROR]"

    async def fine_tune_async(self, training_data: dict, pc_id_override: str = None):
        """Asynchronous placeholder for client-side fine-tuning."""
        current_pc_id = pc_id_override if pc_id_override else self.pc_id
        print(f"Client LLMEngine ({current_pc_id}): Async fine-tuning requested (placeholder).")

        def _save_data_blocking():
            data_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "training_data_local", current_pc_id)
            os.makedirs(data_dir, exist_ok=True)
            timestamp = torch.randint(0, 1000000, (1,)).item()
            data_file = os.path.join(data_dir, f"training_sample_{timestamp}.json")
            try:
                import json
                with open(data_file, "w", encoding="utf-8") as f:
                    json.dump(training_data, f, ensure_ascii=False, indent=4)
                print(f"Client LLMEngine ({current_pc_id}): Saved training data sample to {data_file}")
            except Exception as e:
                print(f"Client LLMEngine ({current_pc_id}): Error saving local training data: {e}")

        await asyncio.to_thread(_save_data_blocking)
        print("Client LLMEngine: Actual fine-tuning process (async) is not yet implemented.")
        # TODO: Implement actual async fine-tuning using asyncio.to_thread for blocking parts like Trainer.train()

    # Kept original fine_tune as synchronous if needed elsewhere, or can be removed.
    def fine_tune(self, training_data: dict, pc_id_override: str = None):
        current_pc_id = pc_id_override if pc_id_override else self.pc_id
        print(f"Client LLMEngine ({current_pc_id}): Fine-tuning requested (SYNC placeholder).")
        data_dir = os.path.join(CLIENT_LLM_MODELS_PATH, "training_data_local", current_pc_id)
        os.makedirs(data_dir, exist_ok=True)
        timestamp = torch.randint(0, 1000000, (1,)).item()
        data_file = os.path.join(data_dir, f"training_sample_sync_{timestamp}.json")
        try:
            import json
            with open(data_file, "w", encoding="utf-8") as f: json.dump(training_data, f, ensure_ascii=False, indent=4)
            print(f"Client LLMEngine ({current_pc_id}): Saved SYNC training data to {data_file}")
        except Exception as e: print(f"Client LLMEngine ({current_pc_id}): Error saving SYNC training data: {e}")
        print("Client LLMEngine: Actual SYNC fine-tuning not implemented.")


    async def save_adapters_async(self):
        """Asynchronously saves LoRA adapters if the model is a PeftModel."""
        if self.model and isinstance(self.model, PeftModel):
            def _save_adapters_blocking():
                self.model.save_pretrained(self.adapters_path)
            try:
                await asyncio.to_thread(_save_adapters_blocking)
                print(f"Client LLMEngine ({self.pc_id}): LoRA adapters saved (async) to {self.adapters_path}")
            except Exception as e:
                print(f"Client LLMEngine ({self.pc_id}): Error saving LoRA adapters (async): {e}")
        else:
            print(f"Client LLMEngine ({self.pc_id}): Model not PeftModel or not init, cannot save adapters (async).")


if __name__ == "__main__":
    async def test_async_llm_engine():
        print("--- Client LLMEngine Async Test ---")
        engine = LLMEngine(pc_id="test_pc_async")
        if engine.is_initialized:
            prompt = "Write a short poem about asynchronous programming."
            print(f"\nTest Prompt: {prompt}")
            response = await engine.generate(prompt, max_new_tokens=60)
            print(f"Async Response: {response}")

            print("\nTesting async fine_tune (placeholder)...")
            await engine.fine_tune_async({"input": "Async test input", "output": "Async test output"})
            # await engine.save_adapters_async() # Test saving (would only work if adapters were loaded/created)
        else:
            print(f"Failed to initialize LLMEngine for async test.")
        print("\n--- Client LLMEngine Async Test Complete ---")

    asyncio.run(test_async_llm_engine())
