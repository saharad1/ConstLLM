#!/usr/bin/env python3
"""
Script for collecting data using selected dataset indices.

This script:
1. Loads selected indices for a specific dataset from a JSON file
2. Filters the original dataset to only include those specific scenario IDs
3. Runs the full data collection pipeline (decision + explanation + attribution) on just those scenarios
4. Saves results in a collection_with_indices specific directory structure
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import wandb
from src.collect_data.base_collector import BaseDataCollector
from src.collect_data.dataset_loader import load_and_prepare_dataset
from src.collect_data.run_collection_utils import load_checkpoints, update_progress


def load_dataset_indices(indices_file: str, dataset_name: str) -> list:
    """Load the selected scenario IDs for the dataset."""
    with open(indices_file, "r") as f:
        data = json.load(f)

    if dataset_name not in data.get("datasets", {}):
        raise ValueError(f"Dataset {dataset_name} not found in indices file")

    selected_indices = data["datasets"][dataset_name].get("selected_indices", [])
    print(f"Loaded {len(selected_indices)} selected indices for {dataset_name}")

    return selected_indices


def filter_dataset_by_indices(dataset, selected_indices: list):
    """
    Filter the dataset to only include scenarios with the selected indices.

    Args:
        dataset: The prepared dataset object
        selected_indices: List of scenario IDs to include

    Returns:
        Filtered dataset containing only the selected scenarios
    """
    # Convert selected_indices to a set for faster lookup
    selected_set = set(selected_indices)

    # Create a list to store filtered scenarios
    filtered_scenarios = []

    print(f"Filtering dataset to include only {len(selected_indices)} selected scenarios...")

    for idx, scenario in enumerate(dataset):
        # Get the scenario ID - this might be the index itself or a specific field
        # For now, we'll use the index as the scenario ID since that's how the split indices work
        if idx in selected_set:
            filtered_scenarios.append(scenario)

    print(f"Filtered dataset contains {len(filtered_scenarios)} scenarios")
    return filtered_scenarios


def setup_collection_with_indices_run_environment(dataset_name, model_id, attribution_method_name, resume_run=None):
    """
    Set up the run environment for data collection with indices.
    Similar to the original but with indices-specific naming.
    """
    # Set up run name and directories
    if resume_run:
        run_name = resume_run
    else:
        current_time = time.strftime("%Y%m%d_%H%M%S")
        # Extract a simplified model name for the run name
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id

        # Create a shorter model identifier for the run name
        if "llama" in model_name.lower():
            # Extract version from names like "Meta-Llama-3.1-8B-Instruct"
            short_model = "llama"  # Default value
            if "-" in model_name:
                parts = model_name.split("-")
                for part in parts:
                    if "." in part and part[0].isdigit():  # Version number like 3.1
                        short_model += part
                        break
        else:
            # For other models, use first 10 chars of model name
            short_model = model_name.split("-")[0][:10]

        run_name = f"{dataset_name}_{current_time}_{attribution_method_name}_{short_model}"

    # Create output directory with indices-specific path
    safe_model_id = model_id.replace("/", "_")
    output_dir = Path(f"data/collect_data_with_indices/{dataset_name}/{safe_model_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create run directory
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up filenames
    jsonl_filename = run_dir / f"{run_name}_results.jsonl"
    checkpoint_file = run_dir / f"{run_name}_checkpoint.json"
    progress_file = run_dir / f"{run_name}_progress.json"

    return run_name, attribution_method_name, run_dir, jsonl_filename, checkpoint_file, progress_file


class IndicesDatasetCollector(BaseDataCollector):
    """Collector for processing dataset with selected indices."""

    def __init__(
        self,
        model_id,
        dataset_name,
        indices_file,
        attribution_method_name,
        num_dec_exp,
        use_wandb,
        resume_run,
        temperature,
        base_seed,
    ):
        super().__init__(
            model_id, dataset_name, attribution_method_name, num_dec_exp, use_wandb, resume_run, temperature, base_seed
        )
        self.indices_file = indices_file
        self.selected_indices = None
        self.filtered_dataset = None

    def setup_run_environment(self):
        """Set up the run environment."""
        run_name, attribution_method_name, output_dir, jsonl_filename, checkpoint_file, progress_file = (
            setup_collection_with_indices_run_environment(
                dataset_name=self.dataset_name,
                model_id=self.model_id,
                attribution_method_name=self.attribution_method_name,
                resume_run=self.resume_run,
            )
        )

        self.output_dir = output_dir
        self.jsonl_filename = jsonl_filename
        self.checkpoint_file = checkpoint_file
        self.progress_file = progress_file

        # Set up wandb if enabled
        if self.use_wandb:
            wandb.init(
                project="constllm_collect_data_with_indices",
                name=run_name,
                config={
                    "model_id": self.model_id,
                    "dataset": self.dataset_name,
                    "attribution_method": self.attribution_method_name,
                    "num_dec_exp": self.num_dec_exp,
                    "temperature": self.temperature,
                    "indices_file": self.indices_file,
                    "num_selected_scenarios": len(self.selected_indices),
                },
            )

    def load_and_filter_dataset(self):
        """Load dataset indices and filter the dataset."""
        # Load dataset indices
        print(f"Loading dataset indices from: {self.indices_file}")
        self.selected_indices = load_dataset_indices(self.indices_file, self.dataset_name)

        # Load full dataset first
        print(f"Loading full dataset: {self.dataset_name}")
        full_dataset = load_and_prepare_dataset(self.dataset_name, subset=None)

        # Filter dataset to only include selected scenario IDs
        self.filtered_dataset = filter_dataset_by_indices(full_dataset, self.selected_indices)

    def run_collection(self, dataset):
        """Run the main collection loop."""
        # Load checkpoints
        self.processed_scenarios, self.progress_data = load_checkpoints(
            checkpoint_file=self.checkpoint_file, progress_file=self.progress_file, total_scenarios=len(dataset)
        )

        # Process scenarios
        for iteration, scenario_item in enumerate(tqdm(dataset, desc="Processing scenarios"), 1):
            # Skip if already processed
            if iteration in self.processed_scenarios:
                print(f"Skipping already processed scenario {iteration}")
                continue

            # Check memory usage
            self.check_memory_usage(iteration)

            # Process the scenario
            scenario_result, scenario_time = self.process_single_scenario_wrapper(scenario_item, iteration)

            if scenario_result:
                # Mark as processed (scenario already saved in process_single_scenario)
                # Note: save_scenario_result is called inside process_single_scenario to handle retries properly
                self.processed_scenarios.add(iteration)

                # Update progress
                failed_scenarios = self.progress_data.get("failed_scenarios", [])
                self.progress_data = update_progress(self.progress_data, self.processed_scenarios, failed_scenarios)

                # Save progress immediately (same as original)
                if hasattr(self, "progress_file") and self.progress_file:
                    try:
                        from src.collect_data.run_collection_utils import save_progress

                        save_progress(self.progress_file, self.progress_data)
                    except Exception as e:
                        print(f"Error saving progress: {e}")

                # Calculate metrics (same order as original)
                metrics = self.calculate_and_log_metrics(scenario_result, iteration, scenario_time)

                # Save checkpoint periodically
                self._save_periodic_checkpoint(iteration)

        # Finalize
        self.print_final_summary(len(dataset))
        self.save_state()
        self.cleanup()


def run_collect_data_with_indices(
    model_id,
    dataset_name,
    indices_file,
    attribution_method_name="LIME",
    num_dec_exp=5,
    use_wandb=True,
    resume_run=None,
    temperature=0.7,
    base_seed=42,
):
    """
    Run the data collection process using selected dataset indices.

    Args:
        model_id: ID of the model to use
        dataset_name: Name of the dataset to use
        indices_file: Path to the dataset indices JSON file
        attribution_method_name: Name of the attribution method to use
        num_dec_exp: Number of decision explanations to generate
        use_wandb: Whether to use wandb for logging
        resume_run: Optional name of a run to resume
        temperature: Temperature for model generation (default: 0.7)
        base_seed: Base seed for reproducible experiments (default: 42)
    """

    # Create collector instance
    collector = IndicesDatasetCollector(
        model_id=model_id,
        dataset_name=dataset_name,
        indices_file=indices_file,
        attribution_method_name=attribution_method_name,
        num_dec_exp=num_dec_exp,
        use_wandb=use_wandb,
        resume_run=resume_run,
        temperature=temperature,
        base_seed=base_seed,
    )

    # Set up the collector
    collector.setup_signal_handlers()
    collector.load_and_filter_dataset()
    collector.setup_run_environment()
    collector.setup_attribution_methods()
    collector.setup_llm_analyzer()
    collector.setup_generation_seeds()
    collector.setup_original_params()

    # Run collection
    collector.run_collection(collector.filtered_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data using selected dataset indices")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--indices_file", type=str, required=True, help="Path to dataset indices JSON file")
    parser.add_argument("--attribution_method", type=str, default="LIME", help="Attribution method to use")
    parser.add_argument("--num_dec_exp", type=int, default=5, help="Number of explanations per decision")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume_run", type=str, help="Name of run to resume")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducible experiments")

    args = parser.parse_args()

    run_collect_data_with_indices(
        model_id=args.model_id,
        dataset_name=args.dataset,
        indices_file=args.indices_file,
        attribution_method_name=args.attribution_method,
        num_dec_exp=args.num_dec_exp,
        use_wandb=not args.no_wandb,
        resume_run=args.resume_run,
        temperature=args.temperature,
        base_seed=args.seed,
    )
