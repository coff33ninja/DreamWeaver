import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset # Using Hugging Face datasets library
import os

class LLMEngine:
    def __init__(self, model_name="TinyLLaMA", db=None):
        self.db = db # Store the database instance
        self.model_name = model_name
        # Define a specific path for PC1's adapters
        # This assumes the LLMEngine instance in CharacterServer is for PC1
        self.adapter_path = f"E:/DreamWeaver/data/models/adapters/{model_name}/PC1"

        # Configure for 4-bit quantization for efficiency (requires bitsandbytes)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

        try:
            # 1. Load base model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto", # Automatically maps model to available devices
                trust_remote_code=True # Needed for some models, e.g., Llama-2
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token # Set pad token for generation

            # Prepare model for k-bit training (e.g., 4-bit)
            self.model = prepare_model_for_kbit_training(self.model)

            # Define LoRA configuration
            lora_config = LoraConfig(
                r=8, # LoRA attention dimension
                lora_alpha=16, # Alpha parameter for LoRA scaling
                target_modules=["q_proj", "v_proj"], # Common modules to apply LoRA to in attention blocks
                lora_dropout=0.05, # Dropout probability for LoRA layers
                bias="none", # Bias type for LoRA layers
                task_type="CAUSAL_LM", # Task type for causal language modeling
            )

            # Try to load existing adapters
            if os.path.exists(self.adapter_path):
                print(f"Loading existing LoRA adapters from {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            else:
                print("No existing adapters found. Initializing new PEFT model.")
                self.model = get_peft_model(self.model, lora_config)

            self.model.print_trainable_parameters() # Print trainable parameters for verification

        except Exception as e:
            print(f"Error loading LLM model {model_name} or configuring PEFT: {e}")
            # Fallback or raise error
            self.model = None
            self.tokenizer = None

    def generate(self, prompt, max_length=100): # Increased max_length for more meaningful responses
        if not self.model or not self.tokenizer:
            return "Error: LLM not loaded."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) # Move inputs to model device
        # Ensure attention_mask is present for generation
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape, dtype=torch.long, device=self.model.device)
        output = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True, # Enable sampling for more varied responses
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id # Important for generation
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip() # .strip() to clean up whitespace

    def fine_tune(self, new_data_point, pc):
        """
        Fine-tunes the LLM using PEFT/LoRA by accumulating all data for the given PC
        from the database.
        """
        if not self.model or not self.tokenizer:
            print("LLM not loaded, cannot fine-tune.")
            return

        if pc != "PC1":
            print(f"Fine-tuning for PC {pc} should happen on the client side. Server LLM only fine-tunes PC1.")
            return

        print(f"Server LLM initiating batch fine-tuning for PC {pc}...")

        # 1. Retrieve all relevant training data for this PC from the database
        # Note: The `new_data_point` passed to this method is already saved to DB
        # by character_server.py before calling fine_tune.
        all_training_data = self.db.get_training_data_for_pc(pc)

        if not all_training_data:
            print(f"No training data found for PC {pc}. Skipping fine-tuning.")
            return

        # 2. Prepare the dataset
        # Combine input and output into a single text string for causal language modeling
        training_examples = [{"text": f"{data['input']}{data['output']}"} for data in all_training_data]

        # Convert to Hugging Face Dataset
        raw_dataset = Dataset.from_list(training_examples)

        # Tokenize the dataset
        def tokenize_function(examples):
            # Ensure max_length is set to a reasonable value, e.g., tokenizer.model_max_length
            # We concatenate input and output, so the model learns to generate output given input.
            return self.tokenizer(examples["text"], truncation=True, max_length=self.tokenizer.model_max_length)

        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # 3. Define TrainingArguments for batch training
        # Adjust batch size and epochs for more substantial training
        batch_size = 4 # Example batch size, adjust based on GPU memory
        gradient_accumulation_steps = 1 # Adjust if batch_size is too small
        num_train_epochs = 3 # Example number of epochs, adjust for desired learning
        training_args = TrainingArguments(
            output_dir=f"./results/{pc}", # Directory to save checkpoints and logs
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=len(tokenized_dataset) // batch_size // 2, # Log twice per epoch
            save_steps=len(tokenized_dataset) // batch_size, # Save once per epoch
            optim="paged_adamw_8bit",
            report_to="none",
            remove_unused_columns=False,
            # For better performance and to avoid warnings, set a default data collator
            # or ensure your dataset is correctly formatted for the Trainer.
            # We'll use DataCollatorForLanguageModeling for causal LM.
        )

        # 4. Create Trainer
        # Use DataCollatorForLanguageModeling to handle padding and masking for causal LM
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # 5. Train the model
        try:
            trainer.train()
            print(f"Fine-tuning completed for PC {pc}.")

            # 6. Save the LoRA adapters
            os.makedirs(self.adapter_path, exist_ok=True)
            self.model.save_pretrained(self.adapter_path)
            print(f"LoRA adapters saved to {self.adapter_path}")

        except Exception as e:
            print(f"Error during fine-tuning for PC {pc}: {e}")
