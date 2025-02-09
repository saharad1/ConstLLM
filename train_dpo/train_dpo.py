# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10)
training_args = DPOConfig(
    output_dir="Qwen2-0.5B-DPO",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10
)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
print("Starting training...")
trainer.train()
print("Training completed.")