import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel

import wandb
from pipeline_dpo.dpo_dataset_codah import load_dpo_dataset
from utils.general import print_memory_usage_all_gpus

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Define model name
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     # quantization_config=bnb_config,  # Apply BitsAndBytes quantization
#     torch_dtype=torch.bfloat16,  # Use FP16 for model weights
#     device_map="auto",
# )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    dtype=torch.bfloat16,
)
model = FastLanguageModel.get_peft_model(
    model=model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)


# model.gradient_checkpointing_enable()

model_device = next(model.parameters()).device
print(f"Model loaded on device: {model_device}")
print_memory_usage_all_gpus("Memory usage after model loading:")

# Ensure padding token exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id


dataset_path = "results/codah_res/codah_results2.jsonl"
train_dataset = load_dpo_dataset(dataset_path, include_scores=False)
print(f"Number of samples: {len(train_dataset)}")
print(train_dataset.column_names)

# Generate a timestamped run name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"Exp-{timestamp}-LLama"
print(f"Run name: {run_name}")

output_dir = (
    Path("trained_models")
    / "codah_models"
    / "LLama-instruct-8b-unsloth-tuned"
    / run_name
)

wandb.init(project="codah-dpo", name=run_name)

# DPO Training Config
training_args = DPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    report_to="wandb",
    per_device_train_batch_size=1,  # Adjust based on available GPU memory
    gradient_accumulation_steps=4,  # Simulates a larger batch size
    num_train_epochs=1,
    learning_rate=5e-6,  # Higher for LoRA
    logging_steps=3,
    save_strategy="epoch",
    beta=0.1,  # Controls DPO optimization strength!
    # bf16=True, # Enable bfloat16 training only when using A100 GPUs
    fp16=True,  # Enable FP16 training only when not using A100 GPUs
    # Evaluation Config
    # eval_strategy="steps",
    # eval_steps=500,
)

# Initialize DPO Trainer
trainer = DPOTrainer(
    model=model,  # Fine-tuned model
    ref_model=None,  # No ref_model for LoRA
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    # peft_config=lora_config,  # Use PEFT
)

# Start training
print("🚀 Starting DPO training with LoRA...")
trainer.train()
print("✅ Training completed.")

# Save the model
save_path = output_dir / "final-model"
trainer.save_model(str(save_path))
print(f"✅ Model saved to {save_path}")
