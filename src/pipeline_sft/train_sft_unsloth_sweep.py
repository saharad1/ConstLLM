from unsloth import FastLanguageModel, is_bfloat16_supported  # isort:skip
import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

import wandb
from src.pipeline_sft.prepare_dataset_to_sft import load_sft_dataset
from src.utils.general import print_gpu_info

print_gpu_info()

MAX_SEQ_LENGTH = 2048


def find_dataset_file(directory: Path, prefix: str) -> Path:
    """Find a file with the given prefix in the directory, regardless of sample count in filename."""
    exact_path = directory / f"{prefix}.jsonl"
    if exact_path.exists():
        return exact_path
    for file_path in directory.glob(f"{prefix}_*.jsonl"):
        return file_path
    raise FileNotFoundError(f"No {prefix} dataset file found in {directory}")


def train_sft_with_config(
    config,
    model_id: str,
    dataset_path: Path,
    run_name: str,
    dataset_name: str,
    similarity_metric: str,
    diff_threshold_train: float = None,
    diff_threshold_eval: float = None,
) -> SFTTrainer:
    """Run a single SFT training with the given config parameters."""
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=MAX_SEQ_LENGTH,
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

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]" if "mistral" in getattr(tokenizer, "name_or_path", "").lower() else "<|pad|>"}
        )
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id

    train_path = find_dataset_file(dataset_path, "train")
    eval_path = find_dataset_file(dataset_path, "eval")

    print(f"Using train dataset: {train_path}")
    print(f"Using eval dataset: {eval_path}")

    train_dataset = load_sft_dataset(
        file_path=str(train_path),
        similarity_metric=similarity_metric,
        diff_threshold=diff_threshold_train,
    )
    eval_dataset = load_sft_dataset(
        file_path=str(eval_path),
        similarity_metric=similarity_metric,
        diff_threshold=diff_threshold_eval,
    )
    print(f"Number of train samples: {len(train_dataset)}")
    print(f"Number of eval samples: {len(eval_dataset)}")

    wandb.config.update(
        {"train_dataset_size": len(train_dataset), "eval_dataset_size": len(eval_dataset)}, allow_val_change=True
    )

    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    output_dir = Path("models") / dataset_name / model_name / run_name

    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to="wandb",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_torch",
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print(
        f"Starting SFT training with {similarity_metric} similarity (diff threshold: {diff_threshold_train} (train), {diff_threshold_eval} (eval))..."
    )
    trainer.train()
    print("Training completed.")

    best_model_path = output_dir / "best_model"
    trainer.save_model(str(best_model_path))
    print(f"Best model (based on {training_args.metric_for_best_model}) saved to {best_model_path}")

    return trainer


def run_sweep(
    dataset_path: Path,
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    dataset_name: str = "ecqa",
    similarity_metric: str = "cosine",
    diff_threshold_train: float = None,
    diff_threshold_eval: float = None,
    sweep_count: int = 10,
):
    """Run a hyperparameter sweep for SFT using wandb."""
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-7, "max": 1e-5, "distribution": "log_uniform_values"},
            "per_device_train_batch_size": {"values": [8, 16]},
            "gradient_accumulation_steps": {"values": [8, 16]},
            "warmup_ratio": {"min": 0.1, "max": 0.2, "distribution": "uniform"},
            "weight_decay": {"min": 0.01, "max": 0.05, "distribution": "uniform"},
            "num_train_epochs": {"values": [10]},
            "lr_scheduler_type": {"values": ["linear", "cosine"]},
        },
    }

    dataset_path_str = str(dataset_path)
    attribution_method = "unknown"
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

    model_short_name = model_id.split("/")[-1] if "/" in model_id else model_id
    sweep_project = f"sft-{model_short_name}-{dataset_name}-{similarity_metric}-{attribution_method}_sweep"

    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_project)

    def sweep_train():
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run = wandb.init()
        config = wandb.config
        run_name = f"{dataset_name}_{timestamp}_lr{config.learning_rate:.2e}"
        run.name = run_name
        run.save()

        wandb.config.update(
            {
                "similarity_metric": similarity_metric,
                "diff_threshold_eval": diff_threshold_eval,
                "dataset": dataset_name,
                "model_id": model_id,
                "attribution_method": attribution_method,
            },
            allow_val_change=True,
        )

        print(f"Using learning rate: {config.learning_rate}")

        train_sft_with_config(
            config=config,
            model_id=model_id,
            dataset_path=dataset_path,
            run_name=run_name,
            dataset_name=dataset_name,
            similarity_metric=similarity_metric,
            diff_threshold_train=diff_threshold_train,
            diff_threshold_eval=diff_threshold_eval,
        )
        wandb.finish()

    print(f"Starting sweep with {sweep_count} runs:")
    print(f"\tDataset: {dataset_name}")
    print(f"\tSimilarity metric: {similarity_metric}")
    print(f"\tAttribution method: {attribution_method}")
    print(f"\tDifference threshold: {diff_threshold_train} (train), {diff_threshold_eval} (eval)")

    wandb.agent(sweep_id, function=sweep_train, count=sweep_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT training sweep with Unsloth")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
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
        help="Threshold for filtering eval examples based on score difference",
    )
    parser.add_argument("--sweep_count", type=int, default=10, help="Number of sweeps to run")

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    run_sweep(
        dataset_path=dataset_path,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        similarity_metric=args.similarity_metric,
        diff_threshold_train=args.diff_threshold_train,
        diff_threshold_eval=args.diff_threshold_eval,
        sweep_count=args.sweep_count,
    )
