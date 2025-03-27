import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

PatchDPOTrainer()
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
    diff_threshold_train: float = None,
    diff_threshold_eval: float = None,
    include_scores=False,
) -> DPOTrainer:
    """Run a single DPO training with the given config parameters."""

    # Log the fixed parameters to WandB:
    wandb.config.update(
        {
            "similarity_metric": similarity_metric,
            "diff_threshold_train": diff_threshold_train,
            "diff_threshold_eval": diff_threshold_eval,
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

    # Load the DPO dataset with the specified similarity metric and threshold
    # Find the train and eval files without relying on specific sample counts
    def find_dataset_file(directory, prefix):
        """Find a file with the given prefix in the directory, regardless of sample count in filename."""
        # First try the exact name format (e.g., 'train.jsonl')
        exact_path = directory / f"{prefix}.jsonl"
        if exact_path.exists():
            return exact_path

        # Then try with sample count (e.g., 'train_7617.jsonl')
        for file_path in directory.glob(f"{prefix}_*.jsonl"):
            return file_path

        # If no matching file found, raise an error
        raise FileNotFoundError(f"No {prefix} dataset file found in {directory}")

    train_path = find_dataset_file(dataset_path, "train")
    eval_path = find_dataset_file(dataset_path, "eval")

    print(f"Using train dataset: {train_path}")
    print(f"Using eval dataset: {eval_path}")

    train_dataset = load_dpo_dataset(
        file_path=str(train_path), similarity_metric=similarity_metric, diff_threshold=diff_threshold_train, include_scores=include_scores
    )
    eval_dataset = load_dpo_dataset(
        file_path=str(eval_path), similarity_metric=similarity_metric, diff_threshold=diff_threshold_eval, include_scores=include_scores
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
        metric_for_best_model="rewards/chosen",  # Use the metric for best model (no need to add eval/)
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
    print(
        f" Starting DPO training with {similarity_metric} similarity (diff threshold: {diff_threshold_train} (train), {diff_threshold_eval} (eval))..."
    )
    trainer.train()
    print(" Training completed.")

    # Save the best model (already loaded due to load_best_model_at_end=True)
    best_model_path = output_dir / "best_model"
    trainer.save_model(str(best_model_path))
    print(f"Best model (based on {training_args.metric_for_best_model}) saved to {best_model_path}")

    return trainer


def run_sweep(
    dataset_path: Path,
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    include_scores=False,
    dataset_name="ecqa",  # ecqa or codah
    similarity_metric="cosine",  # "spearman" or "cosine"
    diff_threshold_train=0.2,
    diff_threshold_eval=0.2,
    sweep_count=10,
):
    """
    Run a hyperparameter sweep using wandb.

    Args:
        model_name: Name of the model to use
        include_scores: Whether to include scores in the training data
        dataset_name: Name of the dataset (codah, ecqa)
        similarity_metric: Which similarity metric to use (spearman, cosine)
        diff_threshold_train: Threshold for filtering examples based on score difference
        diff_threshold_eval: Threshold for filtering examples based on score difference
        sweep_count: Number of sweeps to run
    """
    # Define sweep configuration
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "eval/rewards/chosen", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"min": 1e-7, "max": 5e-6, "distribution": "log_uniform_values"},
            "beta": {"min": 0.05, "max": 0.3, "distribution": "uniform"},
            "per_device_train_batch_size": {"values": [32]},
            "gradient_accumulation_steps": {"values": [16]},
            "warmup_ratio": {"min": 0, "max": 0.1, "distribution": "uniform"},
            "weight_decay": {"min": 0.005, "max": 0.015, "distribution": "uniform"},
            "num_train_epochs": {"values": [20]},
            "lr_scheduler_type": {"values": ["cosine", "linear"]},
        },
    }
    # Create a unique sweep project name that includes model, dataset and similarity metric
    model_short_name = model_name.split("/")[-1]  # Extract just the model name without organization
    sweep_project = f"dpo-{model_short_name}-{dataset_name}-{similarity_metric}_sweep"

    # Include fixed parameters in the sweep name
    sweep_config_name = f"{similarity_metric}_diff{diff_threshold_train}_bs{sweep_config['parameters']['per_device_train_batch_size']['values'][0]}_ep{sweep_config['parameters']['num_train_epochs']['values'][0]}"

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
                "diff_threshold_train": diff_threshold_train,
                "diff_threshold_eval": diff_threshold_eval,
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
            diff_threshold_train=diff_threshold_train,
            diff_threshold_eval=diff_threshold_eval,
            include_scores=include_scores,
        )

        # Finish the run
        wandb.finish()

    # Start the sweep
    print(f"Starting sweep with {sweep_count} runs:")
    print(f"\tDataset: {dataset_name}")
    print(f"\tSimilarity metric: {similarity_metric}")
    print(f"\tDifference threshold: {diff_threshold_train} (train), {diff_threshold_eval} (eval)")

    wandb.agent(sweep_id, function=sweep_train, count=sweep_count)


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Run DPO training sweep with Unsloth")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Name of the model to use")
    parser.add_argument("--dataset_name", type=str, default="ecqa", help="Name of the dataset (e.g., ecqa, codah)")
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine",
        choices=["cosine", "spearman"],
        help="Similarity metric to use (cosine or spearman)",
    )
    parser.add_argument(
        "--diff_threshold_train", type=float, default=0.2, help="Threshold for filtering examples based on score difference"
    )
    parser.add_argument(
        "--diff_threshold_eval", type=float, default=None, help="Threshold for filtering examples based on score difference"
    )
    parser.add_argument("--sweep_count", type=int, default=10, help="Number of sweeps to run")
    parser.add_argument("--include_scores", action="store_true", help="Whether to include scores in the training data")

    args = parser.parse_args()

    # Convert dataset_path to Path object
    dataset_path = Path(args.dataset_path)

    # Run sweep with parsed arguments
    run_sweep(
        dataset_path=dataset_path,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        similarity_metric=args.similarity_metric,
        diff_threshold_train=args.diff_threshold_train,
        diff_threshold_eval=args.diff_threshold_eval,
        sweep_count=args.sweep_count,
        include_scores=args.include_scores,
    )
