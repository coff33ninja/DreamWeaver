import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
import os

class LLMEngine:
    def __init__(self, model_name="TinyLLaMA"):
        # Client-side LLM, potentially a smaller model or quantized version
        self.model_name = model_name

        # Configure for 4-bit quantization for efficiency (requires bitsandbytes)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        try:
            # Load base model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto", # Automatically maps model to available devices
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)

            # Define LoRA configuration
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"], # Common modules for Llama-like models
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # For clients, we don't persist adapters to disk in a shared location.
            # The fine-tuning updates the in-memory model.
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except Exception as e:
            print(f"Error loading LLM model {model_name}: {e}")
            # Fallback or raise error
            self.model = None
            self.tokenizer = None

    def generate(self, prompt, max_length=100):
        if not self.model or not self.tokenizer:
            return "Error: LLM not loaded."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape, dtype=torch.long, device=self.model.device)

        output = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def fine_tune(self, dataset, pc):
        """
        Fine-tunes the client's LLM using PEFT/LoRA with a new data point.
        This updates the in-memory model.
        """
        if not self.model or not self.tokenizer:
            print(f"LLM not loaded for client {pc}, cannot fine-tune.")
            return

        print(f"Client {pc} fine-tuning with new data: {dataset}")

        training_examples = [{"text": f"{dataset['input']}{dataset['output']}"}]
        raw_dataset = Dataset.from_list(training_examples)

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=self.tokenizer.model_max_length)

        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])

        training_args = TrainingArguments(
            output_dir=f"./client_results/{pc}", # Client-side results directory
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            max_steps=1, # Fine-tune on single example
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            save_steps=-1, # Do not save checkpoints on client side
            optim="paged_adamw_8bit",
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )

        try:
            trainer.train()
            print(f"Client {pc} fine-tuning completed.")
        except Exception as e:
            print(f"Error during client-side fine-tuning for PC {pc}: {e}")
