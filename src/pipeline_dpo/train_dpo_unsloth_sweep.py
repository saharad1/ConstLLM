import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

import wandb
from src.pipeline_dpo.prepare_dataset_to_dpo import load_dpo_dataset
from src.utils.general import print_gpu_info

print_gpu_info()


def train_dpo_with_config(
    config,
    model_name: str,
    dataset_path: Path,
    run_name: str,
    dataset_name: str,
    similarity_metric: str,
    diff_threshold: float = 0,
    include_scores=False,
) -> DPOTrainer:
    """Run a single DPO training with the given config parameters."""

    # Log the fixed parameters to WandB:
    wandb.config.update(
        {
            "similarity_metric": similarity_metric,
            "diff_threshold": diff_threshold,
            "dataset": dataset_name,
            "model": model_name,
        },
        allow_val_change=True,
    )

    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load the DPO dataset with the specified similarity metric and threshold
    train_path = dataset_path / "train_7617.jsonl"
    eval_path = dataset_path / "eval_2176.jsonl"
    train_dataset = load_dpo_dataset(
        file_path=str(train_path), similarity_metric=similarity_metric, diff_threshold=diff_threshold, include_scores=include_scores
    )
    eval_dataset = load_dpo_dataset(
        file_path=str(eval_path), similarity_metric=similarity_metric, diff_threshold=None, include_scores=include_scores
    )
    print(f"Number of train samples: {len(train_dataset)}")
    print(f"Number of eval samples: {len(eval_dataset)}")

    # Add dataset sizes to wandb config
    wandb.config.update({"train_dataset_size": len(train_dataset), "eval_dataset_size": len(eval_dataset)}, allow_val_change=True)

    output_dir = Path("trained_models") / f"{dataset_name}_models" / model_name / run_name

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
        metric_for_best_model="rewards/margins",  # Use the metric for best model (no need to add eval/)
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
    print(f" Starting DPO training with {similarity_metric} similarity (diff threshold: {diff_threshold})...")
    trainer.train()
    print(" Training completed.")

    # Save the model
    save_path = output_dir / "final-model"
    trainer.save_model(str(save_path))
    print(f" Model saved to {save_path}")

    return trainer


def run_sweep(
    dataset_path: Path,
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    include_scores=False,
    dataset_name="ecqa",  # ecqa or codah
    similarity_metric="cosine",  # "spearman" or "cosine"
    diff_threshold=0.1,
    sweep_count=10,
):
    """
    Run a hyperparameter sweep using wandb.

    Args:
        model_name: Name of the model to use
        include_scores: Whether to include scores in the training data
        dataset_name: Name of the dataset (codah, ecqa)
        similarity_metric: Which similarity metric to use (spearman, cosine)
        diff_threshold: Threshold for filtering examples based on score difference
        sweep_count: Number of sweeps to run
    """
    # Define sweep configuration
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "eval/rewards/margins", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"min": 1e-5, "max": 8e-5, "distribution": "log_uniform_values"},
            "beta": {"min": 0.08, "max": 0.18, "distribution": "uniform"},
            "per_device_train_batch_size": {"values": [32]},
            "gradient_accumulation_steps": {"values": [16]},
            "warmup_ratio": {"min": 0, "max": 0.1, "distribution": "uniform"},
            "weight_decay": {"min": 0.005, "max": 0.015, "distribution": "uniform"},
            "num_train_epochs": {"values": [30]},
            "lr_scheduler_type": {"values": ["cosine", "linear"]},
        },
    }
    # Create a unique sweep project name that includes model, dataset and similarity metric
    model_short_name = model_name.split("/")[-1]  # Extract just the model name without organization
    sweep_project = f"dpo-{model_short_name}-{dataset_name}-{similarity_metric}_sweep"

    # Include fixed parameters in the sweep name
    sweep_config_name = f"{similarity_metric}_diff{diff_threshold}_bs{sweep_config['parameters']['per_device_train_batch_size']['values'][0]}_ep{sweep_config['parameters']['num_train_epochs']['values'][0]}"

    # Initialize the sweep with WandB
    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_project)

    # Define the sweep training function with fixed parameters captured in closure
    def sweep_train():
        """Inner function that runs a single training with sweep parameters."""
        # Initialize a new wandb run
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

        # Start a new wandb run first
        run = wandb.init()

        # Get the hyperparameters from wandb config
        config = wandb.config

        # Create a more descriptive run name
        run_name = f"{dataset_name}_{similarity_metric}_lr{config.learning_rate:.2e}_beta{config.beta:.2f}_{timestamp}"

        # Update the run name
        run.name = run_name
        run.save()

        # Log fixed parameters explicitly so they appear in the WandB dashboard
        wandb.config.update(
            {
                "similarity_metric": similarity_metric,
                "diff_threshold": diff_threshold,
                "dataset": dataset_name,
                "model": model_name,
            },
            allow_val_change=True,
        )

        # Print the learning rate for debugging
        print(f"Using learning rate: {config.learning_rate}")

        # Run training with these hyperparameters and fixed parameters
        train_dpo_with_config(
            config=config,
            model_name=model_name,
            dataset_path=dataset_path,
            run_name=run_name,  # Use the timestamp run name
            dataset_name=dataset_name,
            similarity_metric=similarity_metric,
            diff_threshold=diff_threshold,
            include_scores=include_scores,
        )

        # Finish the run
        wandb.finish()

    # Start the sweep
    print(f"Starting sweep with {sweep_count} runs:")
    print(f"\tDataset: {dataset_name}")
    print(f"\tSimilarity metric: {similarity_metric}")
    print(f"\tDifference threshold: {diff_threshold}")

    wandb.agent(sweep_id, function=sweep_train, count=sweep_count)


if __name__ == "__main__":
    dataset_path = Path(f"dpo_datasets/cleaned_ecqa_dpo_datasets/cleaned_ecqa_250221_181714_LIME")
    # Example: Run sweep with cosine similarity
    run_sweep(
        dataset_path=dataset_path,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        dataset_name="ecqa",
        similarity_metric="cosine",
        diff_threshold=0.2,
        sweep_count=10,
    )
