import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
from datetime import datetime
from pathlib import Path

import torch

# from peft import LoraConfig, get_peft_model
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel

import wandb

# from datasets import load_dataset
from pipeline_dpo.dpo_dataset_codah import load_dpo_dataset
from utils.general import print_gpu_info

print_gpu_info()


def train_dpo_seq(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", include_scores=False
):
    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
    )

    # model.gradient_checkpointing_enable()

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset_path = (
        Path("dpo_datasets")
        / "cleaned_codah_dpo_datasets"
        / "codah_250213_182318_LLama"
    )
    train_path = dataset_path / "train.jsonl"
    eval_path = dataset_path / "eval.jsonl"
    train_dataset = load_dpo_dataset(str(train_path), include_scores=include_scores)
    eval_dataset = load_dpo_dataset(str(eval_path), include_scores=include_scores)
    print(f"Number of train samples: {len(train_dataset)}")
    print(f"Number of eval samples: {len(eval_dataset)}")

    # Generate a timestamped run name
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"Exp-{timestamp}-LLama"
    print(f"Run name: {run_name}")

    output_dir = (
        Path("trained_models")
        / "codah_models"
        / "LLama-instruct-8b-unsloth-tuned"
        / run_name
    )
    wandb_mode = False
    wandb.init(
        project="codah-train-dpo",
        name=run_name,
        mode="online" if wandb_mode else "disabled",
    )

    # DPO Training Config
    training_args = DPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to="wandb",
        per_device_train_batch_size=32,  # Adjust based on available GPU memory
        gradient_accumulation_steps=8,  # Simulates a larger batch size
        num_train_epochs=10,
        learning_rate=5e-6,  # Higher for LoRA
        logging_steps=10,
        save_strategy="epoch",
        beta=0.1,  # Controls DPO optimization strength!
        bf16=True,  # Enable bfloat16 training only when using A100 GPUs
        # fp16=True,  # Enable FP16 training only when not using A100 GPUs
        # Evaluation Config
        eval_strategy="epoch",
        # eval_steps=20,
    )

    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,  # Fine-tuned model
        ref_model=None,  # No ref_model for LoRA
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        eval_dataset=eval_dataset,  # No evaluation dataset
    )

    # Start training
    print("🚀 Starting DPO training with LoRA...")
    trainer.train()
    print("✅ Training completed.")

    # Save the model
    save_path = output_dir / "final-model"
    trainer.save_model(str(save_path))
    print(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    train_dpo_seq(model_name=model_name, include_scores=False)
