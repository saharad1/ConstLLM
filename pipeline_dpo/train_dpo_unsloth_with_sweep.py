import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

import wandb
from pipeline_dpo.dpo_dataset_codah import load_dpo_dataset
from utils.general import print_gpu_info

print_gpu_info()


def train_dpo_seq(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", include_scores=False) -> None:
    """Original training function with fixed hyperparameters."""
    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load the DPO dataset
    dataset_path = Path("dpo_datasets") / "cleaned_codah_dpo_datasets" / "cleaned_codah_250219_165846_LIME"
    train_path = dataset_path / "train.jsonl"
    eval_path = dataset_path / "eval.jsonl"
    train_dataset = load_dpo_dataset(str(train_path), include_scores=include_scores)
    eval_dataset = load_dpo_dataset(str(eval_path), include_scores=include_scores)
    print(f"Number of train samples: {len(train_dataset)}")
    print(f"Number of eval samples: {len(eval_dataset)}")

    # Generate a timestamped run name
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"codah_{timestamp}_llama"
    print(f"Run name: {run_name}")

    output_dir = Path("trained_models") / "codah_models" / "LLama-instruct-8b" / run_name
    wandb_mode = True
    wandb.init(
        project="tune-llm-dpo",
        name=run_name,
        tags=["codah", "llama"],
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
        bf16=is_bfloat16_supported(),  # Enable bfloat16 training only when using A100 GPUs
        fp16=not is_bfloat16_supported(),  # Enable FP16 training only when not using A100 GPUs
        # Evaluation Config
        eval_strategy="epoch",
        # eval_steps=20,
    )

    PatchDPOTrainer()
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


def train_dpo_with_config(
    config, model_name: str, dataset_path: Path, run_name: str, dataset_name: str, include_scores=False
) -> DPOTrainer:
    """Run a single DPO training with the given config parameters."""

    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load the DPO dataset
    train_path = dataset_path / "train.jsonl"
    eval_path = dataset_path / "eval.jsonl"
    train_dataset = load_dpo_dataset(str(train_path), include_scores=include_scores)
    eval_dataset = load_dpo_dataset(str(eval_path), include_scores=include_scores)
    print(f"Number of train samples: {len(train_dataset)}")
    print(f"Number of eval samples: {len(eval_dataset)}")

    output_dir = Path("trained_models") / f"{dataset_name}_models" / "LLama-instruct-8b" / run_name

    # Create DPO Training Config from sweep parameters
    training_args = DPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to="wandb",
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        beta=config.beta,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_torch",
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rewards/margins",
        greater_is_better=True,
    )

    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Start training
    print("🚀 Starting DPO training with LoRA...")
    trainer.train()
    print("✅ Training completed.")

    # Save the model
    save_path = output_dir / "final-model"
    trainer.save_model(str(save_path))
    print(f"✅ Model saved to {save_path}")

    return trainer


def run_sweep(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", include_scores=False):
    """Run a hyperparameter sweep using wandb."""
    # Define sweep configuration
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "eval/rewards/margins", "goal": "maximize"},
        "parameters": {
            # "learning_rate": {"min": 1e-6, "max": 1e-5, "distribution": "log_uniform_values"},
            "learning_rate": {"values": [1e-6, 2e-6, 4e-6, 5e-6]},
            "beta": {"values": [0.05, 0.1, 0.2]},
            "per_device_train_batch_size": {"values": [16, 32]},
            "gradient_accumulation_steps": {"values": [4, 8, 16]},
            "warmup_ratio": {"values": [0.0, 0.05, 0.1]},
            "weight_decay": {"min": 0.0, "max": 0.1, "distribution": "uniform"},
            "num_train_epochs": {"values": [5, 10, 15]},
            "lr_scheduler_type": {"values": ["cosine", "linear"]},
        },
    }

    # Set dataset
    dataset_name = "codah"  # ecqa or codah
    # Define the dataset path
    dataset_path = Path("dpo_datasets/cleaned_codah_dpo_datasets/cleaned_codah_250219_165846_LIME")

    sweep_id = wandb.sweep(sweep=sweep_config, project=f"dpo-{dataset_name}-sweep")

    # Define the sweep training function
    def sweep_train():
        # Initialize a new wandb run
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_name = f"{dataset_name}_{timestamp}_sweep_{sweep_id}"
        run = wandb.init(name=run_name)

        # Get the hyperparameters from wandb
        config = wandb.config

        # Run training with these hyperparameters
        train_dpo_with_config(
            config=config,
            model_name=model_name,
            dataset_path=dataset_path,
            run_name=dataset_name,
            dataset_name=dataset_name,
            include_scores=include_scores,
        )

        # Finish the run
        wandb.finish()

    # Start the sweep
    wandb.agent(sweep_id, function=sweep_train, count=4)  # count = Run x sweep iterations


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"

    # Choose which mode to run:
    mode = "sweep"  # "train" or "sweep"

    if mode == "train":
        train_dpo_seq(model_name=model_name, include_scores=False)
    elif mode == "sweep":
        run_sweep(model_name=model_name, include_scores=False)
    else:
        print(f"Unknown mode: {mode}. Use 'train' or 'sweep'")
