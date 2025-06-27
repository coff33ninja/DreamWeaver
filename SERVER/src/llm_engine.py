import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel # Added PeftModel
from datasets import Dataset # Using Hugging Face datasets library
import os
from .config import ADAPTERS_PATH # Import from config

class LLMEngine:
    def __init__(self, model_name="TinyLLaMA", db=None): # model_name can be made configurable
        self.db = db
        self.model_name = model_name # e.g., "TinyLLaMA", "NousResearch/Llama-2-7b-chat-hf"

        # Use ADAPTERS_PATH from config and append model-specific and PC-specific parts
        # This LLM Engine on the server is specifically for "PC1"
        self.adapter_path = os.path.join(ADAPTERS_PATH, self.model_name.replace("/", "_"), "PC1") # Sanitize model_name for path
        os.makedirs(self.adapter_path, exist_ok=True) # Ensure adapter directory exists

        # Configure for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for compute if available
            bnb_4bit_use_double_quant=False,
        )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto", # Automatically map model to available devices (CPU, GPU)
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token # Common practice

            self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=16, # Increased rank
                lora_alpha=32, # Increased alpha
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # More target modules for Llama
                lora_dropout=0.1, # Increased dropout
                bias="none",
                task_type="CAUSAL_LM",
            )

            if os.path.exists(os.path.join(self.adapter_path, "adapter_config.json")): # Check for a specific file
                print(f"Loading existing LoRA adapters from {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            else:
                print(f"No existing adapters found at {self.adapter_path}. Initializing new PEFT model.")
                self.model = get_peft_model(self.model, lora_config)

            self.model.print_trainable_parameters()

        except Exception as e:
            print(f"Error loading LLM model {self.model_name} or configuring PEFT: {e}")
            self.model = None
            self.tokenizer = None

    def generate(self, prompt, max_new_tokens=150): # Changed to max_new_tokens
        if not self.model or not self.tokenizer:
            return "Error: LLM not loaded."

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.tokenizer.model_max_length - max_new_tokens).to(self.model.device)

        if "attention_mask" not in inputs:
             inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape, dtype=torch.long, device=self.model.device)

        with torch.no_grad(): # Ensure no gradients are computed during generation
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens, # Controls length of generated text
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the newly generated tokens
        generated_text = self.tokenizer.decode(output_sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return generated_text.strip()


    def fine_tune(self, new_data_point, pc_id):
        if not self.model or not self.tokenizer:
            print("LLM not loaded, cannot fine-tune.")
            return

        if pc_id != "PC1": # Server LLM only fine-tunes for PC1
            print(f"Fine-tuning for PC {pc_id} is handled by the client. Server LLM only manages PC1.")
            return

        print(f"Server LLM initiating fine-tuning for {pc_id} using data point: {new_data_point}")

        # For simplicity in this example, we'll fine-tune on the single new data point.
        # In a more robust system, you'd accumulate data or use a dataset.
        # The `new_data_point` should be a dictionary like {"input": "...", "output": "..."}

        # Combine input and output into a single text string for causal language modeling
        # This format is typical for text-generation tasks.
        text_data = f"{new_data_point['input']}{self.tokenizer.eos_token}{new_data_point['output']}{self.tokenizer.eos_token}"

        # Create a dataset from this single example
        # While not ideal for actual fine-tuning, it demonstrates the process.
        # For real fine-tuning, use a larger dataset loaded via self.db.get_training_data_for_pc(pc_id)
        dataset = Dataset.from_list([{"text": text_data}])

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.tokenizer.model_max_length or 512)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Define TrainingArguments
        # These arguments would typically be more extensive for real training.
        training_args = TrainingArguments(
            output_dir=os.path.join("./results", pc_id), # Temporary directory for logs/checkpoints
            per_device_train_batch_size=1, # Using 1 as we have a single data point for this example
            num_train_epochs=1, # Train for 1 epoch on this single data point
            learning_rate=5e-5, # A common learning rate for fine-tuning
            fp16=True, # Use mixed precision if available
            logging_steps=1,
            report_to="none", # Disable external reporting like wandb for this simple case
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model, # The PEFT model
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        try:
            print(f"Starting fine-tuning for {pc_id}...")
            trainer.train()
            print(f"Fine-tuning completed for {pc_id}.")

            # Save the updated LoRA adapters
            self.model.save_pretrained(self.adapter_path)
            print(f"LoRA adapters for {pc_id} saved to {self.adapter_path}")

        except Exception as e:
            print(f"Error during fine-tuning for {pc_id}: {e}")

    # Example: Method to load all training data for PC1 for a batch fine-tuning session
    def batch_fine_tune_pc1(self):
        if not self.db:
            print("Database not available for batch fine-tuning.")
            return

        all_training_data = self.db.get_training_data_for_pc("PC1")
        if not all_training_data:
            print("No training data found for PC1 for batch fine-tuning.")
            return

        print(f"Starting batch fine-tuning for PC1 with {len(all_training_data)} data points.")

        # Prepare dataset (text format: "input<eos>output<eos>")
        formatted_texts = [f"{data['input']}{self.tokenizer.eos_token}{data['output']}{self.tokenizer.eos_token}" for data in all_training_data]
        dataset = Dataset.from_list([{"text": text} for text in formatted_texts])

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.tokenizer.model_max_length or 512)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        training_args = TrainingArguments(
            output_dir=os.path.join("./results_batch", "PC1"),
            per_device_train_batch_size=max(1, 8 // torch.cuda.device_count() if torch.cuda.is_available() else 1), # Adjust based on VRAM
            gradient_accumulation_steps=2, # Accumulate gradients if batch size is small
            num_train_epochs=3, # Number of epochs for batch training
            learning_rate=2e-4,
            fp16=True, # nvidia-amp
            logging_steps=max(1, len(tokenized_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) // 10), # Log 10 times per epoch
            save_steps=max(1, len(tokenized_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)), # Save once per epoch
            optim="paged_adamw_8bit", # From QLoRA paper
            report_to="none",
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        try:
            trainer.train()
            print("Batch fine-tuning for PC1 completed.")
            self.model.save_pretrained(self.adapter_path)
            print(f"LoRA adapters for PC1 saved to {self.adapter_path} after batch training.")
        except Exception as e:
            print(f"Error during batch fine-tuning for PC1: {e}")
