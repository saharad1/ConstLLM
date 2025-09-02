from unsloth import FastLanguageModel, is_bfloat16_supported  # isort:skip
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, TrainerCallback
from trl import DPOConfig, DPOTrainer

# PatchDPOTrainer()
import wandb
from src.pipeline_dpo.prepare_dataset_to_dpo import load_dpo_dataset
from src.utils.general import print_gpu_info

print_gpu_info()


def train_dpo_with_config(
    config,
    model_id: str,
    dataset_path: Path,
    run_name: str,
    dataset_name: str,
    similarity_metric: str,
    diff_threshold_train: float = None,
    diff_threshold_eval: float = None,
    score_scale_factor: float = 1.0,
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
            "model_id": model_id,
        },
        allow_val_change=True,
    )

    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]" if "mistral" in getattr(tokenizer, "name_or_path", "").lower() else "<|pad|>"}
        )
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id

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
        file_path=str(train_path),
        similarity_metric=similarity_metric,
        diff_threshold=diff_threshold_train,
        include_scores=include_scores,
        score_scale_factor=score_scale_factor,
    )
    eval_dataset = load_dpo_dataset(
        file_path=str(eval_path),
        similarity_metric=similarity_metric,
        diff_threshold=diff_threshold_eval,
        include_scores=include_scores,
        score_scale_factor=score_scale_factor,
    )
    print(f"Number of train samples: {len(train_dataset)}")
    print(f"Number of eval samples: {len(eval_dataset)}")

    # Add dataset sizes to wandb config
    wandb.config.update(
        {"train_dataset_size": len(train_dataset), "eval_dataset_size": len(eval_dataset)}, allow_val_change=True
    )

    # Extract a simplified model name for the run name
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id

    output_dir = Path("models") / dataset_name / model_name / run_name

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
        save_total_limit=2,
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
        ref_model=None,  # Use PEFT's reference model mechanism
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Add a custom callback to compute and log the combined metric
    class CombinedMetricCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return

            # Print the available metrics for debugging
            print(f"Available metrics: {list(metrics.keys())}")

            # Extract reward and margin metrics - these metrics are provided by the DPO trainer
            # Transformers stores them with eval_ prefix internally
            eval_reward_chosen = metrics.get("eval_rewards/chosen", 0)
            eval_reward_margins = metrics.get("eval_rewards/margins", 0)

            # Compute combined metric with fixed weights
            # eval_combined_metric = 0.5 * eval_reward_chosen + 0.5 * eval_reward_margins
            eval_combined_metric = eval_reward_margins

            # Store the combined metric in the metrics with the same format as other metrics in Transformers
            metrics["eval_combined_metric"] = eval_combined_metric
            # When logging to wandb, use the slash format
            wandb.log({"eval/combined_metric": eval_combined_metric})

    # Add the custom callback to the trainer
    trainer.add_callback(CombinedMetricCallback())

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
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    include_scores=False,
    dataset_name="ecqa",  # ecqa or codah
    similarity_metric="cosine",  # "spearman" or "cosine"
    diff_threshold_train=None,
    diff_threshold_eval=None,
    score_scale_factor=1.0,
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
        "metric": {"name": "eval/combined_metric", "goal": "maximize"},  # Keep the eval/ prefix for wandb
        "parameters": {
            "learning_rate": {"min": 1e-7, "max": 1e-5, "distribution": "log_uniform_values"},
            "beta": {"min": 3, "max": 10, "distribution": "uniform"},
            "per_device_train_batch_size": {"values": [8, 16]},  # Reduced from 32
            "gradient_accumulation_steps": {"values": [8, 16]},  # Increased from 4
            "warmup_ratio": {"min": 0.1, "max": 0.2, "distribution": "uniform"},
            "weight_decay": {"min": 0.01, "max": 0.05, "distribution": "uniform"},
            "num_train_epochs": {"values": [10]},
            "lr_scheduler_type": {"values": ["linear", "cosine"]},
            # "diff_threshold_train": {"min": 0, "max": 0.2, "distribution": "uniform"},
        },
    }

    # sweep_config = {
    #     "method": "bayes",  # Bayesian optimization
    #     "metric": {"name": "eval/combined_metric", "goal": "maximize"},  # Keep the eval/ prefix for wandb
    #     "parameters": {
    #         "learning_rate": {"min": 8e-7, "max": 5e-6, "distribution": "log_uniform_values"},
    #         "beta": {"min": 3, "max": 7, "distribution": "uniform"},
    #         "per_device_train_batch_size": {"values": [8, 16]},  # Reduced from 32
    #         "gradient_accumulation_steps": {"values": [8, 16]},  # Increased from 4
    #         "warmup_ratio": {"min": 0.1, "max": 0.2, "distribution": "uniform"},
    #         "weight_decay": {"min": 0.01, "max": 0.05, "distribution": "uniform"},
    #         "num_train_epochs": {"values": [10]},
    #         "lr_scheduler_type": {"values": ["linear"]},
    #         # "diff_threshold_train": {"min": 0, "max": 0.2, "distribution": "uniform"},
    #     },
    # }
    # Extract attribution method from dataset path
    dataset_path_str = str(dataset_path)
    attribution_method = "unknown"

    # Look for attribution method in the dataset path
    if "_LIME_" in dataset_path_str:
        attribution_method = "LIME"
    elif "_LIG_" in dataset_path_str:
        attribution_method = "LIG"
    elif "_SHAPLEY_VALUE_SAMPLING_" in dataset_path_str:
        attribution_method = "SHAPLEY_VALUE_SAMPLING"
    elif "_FEATURE_ABLATION_" in dataset_path_str:
        attribution_method = "FEATURE_ABLATION"
    elif "_KSHAP_" in dataset_path_str:
        attribution_method = "KSHAP"

    # Create a unique sweep project name that includes model, dataset, similarity metric, and attribution method
    model_short_name = model_id.split("/")[-1] if "/" in model_id else model_id  # Extract just the model name
    sweep_project = f"dpo-{model_short_name}-{dataset_name}-{similarity_metric}-{attribution_method}_sweep"

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
        run_name = f"{dataset_name}_{timestamp}_lr{config.learning_rate:.2e}_beta{config.beta:.2f}"

        # Update the run name
        run.name = run_name
        run.save()

        # Log fixed parameters explicitly so they appear in the WandB dashboard
        wandb.config.update(
            {
                "similarity_metric": similarity_metric,
                # "diff_threshold_train": diff_threshold_train,
                "diff_threshold_eval": diff_threshold_eval,
                "dataset": dataset_name,
                "model_id": model_id,
                "score_scale_factor": score_scale_factor,
                "attribution_method": attribution_method,
            },
            allow_val_change=True,
        )

        # Print the learning rate for debugging
        print(f"Using learning rate: {config.learning_rate}")

        # Run training with these hyperparameters and fixed parameters
        train_dpo_with_config(
            config=config,
            model_id=model_id,
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
    print(f"\tAttribution method: {attribution_method}")
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
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Name of the model to use"
    )
    parser.add_argument("--dataset_name", type=str, default="ecqa", help="Name of the dataset (e.g., ecqa, codah)")
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine",
        choices=["cosine", "spearman"],
        help="Similarity metric to use (cosine or spearman)",
    )
    parser.add_argument(
        "--diff_threshold_train",
        type=float,
        default=0,
        help="Threshold for filtering examples based on score difference",
    )
    parser.add_argument(
        "--diff_threshold_eval",
        type=float,
        default=None,
        help="Threshold for filtering examples based on score difference",
    )
    parser.add_argument("--sweep_count", type=int, default=10, help="Number of sweeps to run")
    parser.add_argument("--include_scores", action="store_true", help="Whether to include scores in the training data")
    parser.add_argument("--score_scale_factor", type=float, default=100.0, help="Scale factor for scores")

    args = parser.parse_args()

    # Convert dataset_path to Path object
    dataset_path = Path(args.dataset_path)

    # Run sweep with parsed arguments
    run_sweep(
        dataset_path=dataset_path,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        similarity_metric=args.similarity_metric,
        diff_threshold_train=args.diff_threshold_train,
        diff_threshold_eval=args.diff_threshold_eval,
        sweep_count=args.sweep_count,
        include_scores=args.include_scores,
        score_scale_factor=args.score_scale_factor,
    )
