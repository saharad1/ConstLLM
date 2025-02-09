import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def print_memory_usage_all_gpus(message=""):
    """Prints the GPU memory usage for all available GPUs in MB"""
    print(f"\n{message}")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)  # Switch to GPU i
        allocated = torch.cuda.memory_allocated(i) / 1024**2  # Convert bytes to MB
        reserved = torch.cuda.memory_reserved(i) / 1024**2  # Convert bytes to MB
        print(f"GPU {i}: Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

# Define model name
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4-bit Quantization Config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Use 4-bit quantization
#     bnb_4bit_compute_dtype=torch.float16,  # Reduce precision
#     bnb_4bit_use_double_quant=True,  # Nested quantization
#     bnb_4bit_quant_type="nf4",  # Normalized float4
# )

# 8-bit Quantization Config
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # Enable 8-bit quantization
#     llm_int8_has_fp16_weight=False,  # Disable FP16 weight quantization
# )

# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,  # Apply BitsAndBytes quantization
    torch_dtype=torch.bfloat16,  # Use FP16 for model weights
    device_map="auto",
)

model.gradient_checkpointing_enable()

model_device = next(model.parameters()).device
print(f"Model loaded on device: {model_device}")
print_memory_usage_all_gpus("Memory usage after model loading:")

# Ensure padding token exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id


# LoRA Configuration
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,  # Dropout to prevent overfitting
)

# Apply LoRA to the base model
model = get_peft_model(model, lora_config)


# Load dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
# dataset = dataset.select(range(10))
print(dataset[0])  # Check the first sample

# Ensure correct dataset columns for DPO
# dataset = dataset.rename_columns({
#     "question": "prompt",  # 🚀 FIXED!
#     "better_response": "chosen",
#     "worse_response": "rejected"
# })

# DPO Training Config
training_args = DPOConfig(
    output_dir="LLama-Tuned",
    run_name="LLama-DPO-Experiment",
    report_to="wandb",
    per_device_train_batch_size=10,  # Adjust based on available GPU memory
    gradient_accumulation_steps=8,  # Simulates a larger batch size
    num_train_epochs=1,
    learning_rate=5e-6,  # Higher for LoRA
    logging_steps=100,
    save_strategy="epoch",
    beta=0.1,  # Controls DPO optimization strength!
    # Evaluation Config
    # eval_strategy="steps",
    # eval_steps=500,
)

# Initialize DPO Trainer
trainer = DPOTrainer(
    model=model,  # Fine-tuned model
    ref_model=None,  # 🚀 FIXED! No ref_model for LoRA
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,  # Use PEFT
)

# Start training
print("🚀 Starting DPO training with LoRA...")
trainer.train()
print("✅ Training completed.")

# Save the model
trainer.save_model("LLama-Tuned-model")
print("✅ Model saved to Qwen2-0.5B-DPO")
