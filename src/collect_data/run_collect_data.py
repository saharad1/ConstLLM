#!/usr/bin/env python3
"""
Script for collecting data by running models on scenarios and computing attributions.
Refactored to use the BaseDataCollector class.
"""

import argparse

from tqdm import tqdm

import wandb
from src.collect_data.base_collector import BaseDataCollector
from src.collect_data.dataset_loader import load_and_prepare_dataset
from src.collect_data.run_collection_utils import (
    load_checkpoints,
    setup_run_environment,
    update_progress,
)


def get_scenario_attribute(scenario_item, dict_key, obj_attr, default=""):
    """
    Helper function to get an attribute from either a dictionary or an object.

    Args:
        scenario_item: Either a dictionary or an object
        dict_key: The key to use if scenario_item is a dictionary
        obj_attr: The attribute name to use if scenario_item is an object
        default: Default value to return if attribute is not found

    Returns:
        The value from the scenario_item
    """
    if isinstance(scenario_item, dict):
        return scenario_item.get(dict_key, default)
    else:
        return getattr(scenario_item, obj_attr, default)


class FullDatasetCollector(BaseDataCollector):
    """Collector for processing the full dataset."""

    def __init__(
        self,
        model_id,
        dataset_name,
        attribution_method_name,
        num_dec_exp,
        subset,
        use_wandb,
        resume_run,
        temperature,
        base_seed,
    ):
        super().__init__(
            model_id, dataset_name, attribution_method_name, num_dec_exp, use_wandb, resume_run, temperature, base_seed
        )
        self.subset = subset

    def setup_run_environment(self):
        """Set up the run environment."""
        run_name, attribution_method_name, output_dir, jsonl_filename, checkpoint_file, progress_file = (
            setup_run_environment(
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
                project="constllm_collect_data",
                name=run_name,
                config={
                    "model_id": self.model_id,
                    "dataset": self.dataset_name,
                    "attribution_method": self.attribution_method_name,
                    "num_dec_exp": self.num_dec_exp,
                    "temperature": self.temperature,
                },
            )

    def run_collection(self, dataset):
        """Run the main collection loop."""
        # Load checkpoints
        self.processed_scenarios, self.progress_data = load_checkpoints(
            checkpoint_file=self.checkpoint_file, progress_file=self.progress_file, total_scenarios=len(dataset)
        )

        # Process scenarios
        for iteration, scenario_item in enumerate(tqdm(dataset, desc="Processing scenarios"), 1):
            # Skip already processed scenarios
            scenario_id = get_scenario_attribute(scenario_item, "scenario_id", "scenario_id", f"scenario_{iteration}")
            if scenario_id in self.processed_scenarios:
                continue

            # Check memory usage
            self.check_memory_usage(iteration)

            # Process the scenario
            scenario_result, scenario_time = self.process_single_scenario_wrapper(scenario_item, iteration, scenario_id)

            if scenario_result:
                # Mark as processed (scenario already saved in process_single_scenario)
                # Note: save_scenario_result is called inside process_single_scenario to handle retries properly
                self.processed_scenarios.add(scenario_id)

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
        print(f"Data collection completed. Processed {len(self.processed_scenarios)} scenarios.")
        self.save_state()
        self.cleanup()


def run_collect_data(
    model_id,
    dataset_name,
    attribution_method_name="LIME",
    num_dec_exp=5,
    subset=None,
    use_wandb=True,
    resume_run=None,
    temperature=0.7,
    base_seed=42,
):
    """
    Run the data collection process for a given model and dataset.

    Args:
        model_id: ID of the model to use
        dataset_name: Name of the dataset to use
        attribution_method_name: Name of the attribution method to use
        num_dec_exp: Number of decision explanations to generate
        subset: Optional subset size of the dataset to use
        use_wandb: Whether to use wandb for logging
        resume_run: Optional name of a run to resume
        temperature: Temperature for model generation (default: 0.7)
        base_seed: Base seed for reproducible experiments (default: 42)
    """

    # Create collector instance
    collector = FullDatasetCollector(
        model_id=model_id,
        dataset_name=dataset_name,
        attribution_method_name=attribution_method_name,
        num_dec_exp=num_dec_exp,
        subset=subset,
        use_wandb=use_wandb,
        resume_run=resume_run,
        temperature=temperature,
        base_seed=base_seed,
    )

    # Set up the collector
    collector.setup_signal_handlers()
    collector.setup_run_environment()
    collector.setup_attribution_methods()
    collector.setup_llm_analyzer()
    collector.setup_generation_seeds()
    collector.setup_original_params()

    # Load dataset and run collection
    dataset = load_and_prepare_dataset(dataset_name, subset)
    collector.run_collection(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data using a specified model")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use")
    parser.add_argument("--dataset", type=str, default="ecqa", help="Dataset to use")
    parser.add_argument("--attribution_method", type=str, default="LIME", help="Attribution method to use")
    parser.add_argument("--num_dec_exp", type=int, default=5, help="Number of explanations per decision")
    parser.add_argument("--subset", type=int, default=None, help="Size of dataset subset to use")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume_run", type=str, help="Name of run to resume")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducible experiments")

    args = parser.parse_args()

    run_collect_data(
        model_id=args.model_id,
        dataset_name=args.dataset,
        attribution_method_name=args.attribution_method,
        num_dec_exp=args.num_dec_exp,
        subset=args.subset,
        use_wandb=not args.no_wandb,
        resume_run=args.resume_run,
        temperature=args.temperature,
        base_seed=args.seed,
    )
