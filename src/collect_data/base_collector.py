#!/usr/bin/env python3
"""
Base class for data collection scripts.
Contains all the common logic for processing scenarios, calculating metrics, and logging.
"""

import signal
import sys
import time
from pathlib import Path

import psutil
import torch

import wandb
from src.collect_data.attribution_config import get_attribution_methods_params
from src.collect_data.collection_metrics import calculate_metrics
from src.collect_data.run_collection_utils import (
    save_checkpoint,
    save_progress,
    update_progress,
)
from src.collect_data.scenario_runner import (
    process_single_scenario,
    save_scenario_details,
)
from src.llm_attribution.LLMAnalyzer import LLMAnalyzer


class BaseDataCollector:
    """Base class for data collection with common functionality."""

    # Constants
    MEMORY_CHECK_INTERVAL = 20
    RELOAD_MODEL_EVERY = 50

    def __init__(
        self,
        model_id,
        dataset_name,
        attribution_method_name,
        num_dec_exp,
        use_wandb,
        resume_run,
        temperature,
        base_seed,
    ):
        """Initialize the base collector."""
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.attribution_method_name = attribution_method_name
        self.num_dec_exp = num_dec_exp
        self.use_wandb = use_wandb
        self.resume_run = resume_run
        self.temperature = temperature
        self.base_seed = base_seed

        # Initialize tracking variables
        self.last_model_reload = 0
        self.last_memory_check = 0
        self.success_sum = 0
        self.spearman_sums = {"best": 0, "worst": 0, "median": 0}
        self.cosine_sums = {"best": 0, "worst": 0, "median": 0}
        self.total_time_sum = 0

        # Will be set by subclasses
        self.output_dir = None
        self.jsonl_filename = None
        self.checkpoint_file = None
        self.progress_file = None
        self.processed_scenarios = set()
        self.progress_data = {}
        self.llm_analyzer = None
        self.methods_params_decision = None
        self.methods_params_explanation = None
        self.generation_seeds = None
        self.original_params = None

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful termination."""

        def signal_handler(sig, frame):
            print("\nReceived termination signal. Saving checkpoint and exiting...")
            self.save_state()
            if self.use_wandb:
                wandb.finish()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def setup_attribution_methods(self):
        """Set up attribution methods parameters."""
        self.methods_params_decision, self.methods_params_explanation = get_attribution_methods_params(
            self.attribution_method_name
        )

    def setup_llm_analyzer(self):
        """Initialize the LLM analyzer."""
        print(f"Loading model: {self.model_id}")
        self.llm_analyzer = LLMAnalyzer(model_id=self.model_id, temperature=self.temperature)

    def setup_generation_seeds(self):
        """Set up deterministic seeds for generation."""
        print(f"Using base seed: {self.base_seed}")
        self.generation_seeds = [self.base_seed + i for i in range(self.num_dec_exp)]
        print(f"Generated seeds for explanations: {self.generation_seeds}")

    def setup_original_params(self):
        """Store original parameters for reset after retries."""
        self.original_params = {
            "model_id": self.model_id,
            "decision": self.methods_params_decision.copy(),
            "explanation": self.methods_params_explanation.copy(),
        }

    def check_memory_usage(self, iteration):
        """Check memory usage and reload model if needed."""
        if iteration - self.last_memory_check >= self.MEMORY_CHECK_INTERVAL:
            memory_usage = psutil.virtual_memory().percent
            print(f"Memory usage: {memory_usage:.1f}%")
            self.last_memory_check = iteration

            if memory_usage > 90:
                print(f"High memory usage detected ({memory_usage}%). Saving checkpoint and exiting...")
                self.save_state()
                if self.use_wandb:
                    wandb.finish()
                sys.exit(0)

        if iteration - self.last_model_reload >= self.RELOAD_MODEL_EVERY:
            print("Reloading model to free memory...")
            del self.llm_analyzer
            torch.cuda.empty_cache()
            time.sleep(2)  # Give some time for memory to be freed
            self.llm_analyzer = LLMAnalyzer(model_id=self.model_id, temperature=self.temperature)
            self.last_model_reload = iteration

    def process_single_scenario_wrapper(self, scenario_item, iteration, scenario_id=None):
        """Process a single scenario with error handling."""
        start_time = time.time()
        try:
            scenario_result, error_msg = process_single_scenario(
                scenario_item=scenario_item,
                llm_analyzer=self.llm_analyzer,
                methods_params_decision=self.methods_params_explanation,
                methods_params_explanation=self.methods_params_explanation,
                generation_seeds=self.generation_seeds,
                num_dec_exp=self.num_dec_exp,
                original_params=self.original_params,
                iteration=iteration,
                output_dir=self.output_dir,
                jsonl_filename=self.jsonl_filename,
            )

            # Add timing calculation (same as original)
            scenario_time = time.time() - start_time
            self.total_time_sum += scenario_time

            # Skip if processing failed (same as original)
            if scenario_result is None:
                # Handle failed scenario
                self._handle_failed_scenario(iteration, error_msg, scenario_id)
                return None, scenario_time

            # Save scenario details (same as original)
            if hasattr(self, "output_dir") and self.output_dir:
                try:
                    output_path = Path(self.output_dir) if isinstance(self.output_dir, str) else self.output_dir
                    save_scenario_details(scenario_result, output_path)
                except Exception as e:
                    print(f"Error saving scenario details: {e}")

            # Note: save_scenario_result is called inside process_single_scenario to handle retries properly
            return scenario_result, scenario_time

        except Exception as e:
            scenario_time = time.time() - start_time
            self.total_time_sum += scenario_time
            print(f"Error processing scenario {iteration}: {e}")
            self._handle_failed_scenario(iteration, str(e), scenario_id)
            return None, scenario_time

    def calculate_and_log_metrics(self, scenario_result, iteration, scenario_time):
        """Calculate metrics and log to wandb (separate from processing)."""
        # Calculate metrics (same order as original)
        metrics, self.success_sum = calculate_metrics(
            scenario_res=scenario_result,
            success_sum=self.success_sum,
            iteration=iteration,
            spearman_sums=self.spearman_sums,
            cosine_sums=self.cosine_sums,
            scenario_time=scenario_time,
            total_time_sum=self.total_time_sum,
        )

        # Log metrics to wandb (same as original)
        if self.use_wandb:
            wandb.log(metrics)

        return metrics

    def _handle_failed_scenario(self, iteration, error_msg, scenario_id=None):
        """Handle a failed scenario by logging error and updating progress."""
        # Log error to file
        if hasattr(self, "output_dir") and self.output_dir:
            try:
                # Ensure output_dir is a Path object
                output_path = Path(self.output_dir) if isinstance(self.output_dir, str) else self.output_dir
                error_file = output_path / "errors.log"

                # Use the same error format as the original
                if scenario_id:
                    error_line = f"Scenario {scenario_id} (iteration {iteration}): {error_msg}\n"
                else:
                    error_line = f"Scenario {iteration}: {error_msg}\n"

                with open(error_file, "a") as f:
                    f.write(error_line)
            except Exception as e:
                print(f"Error writing to error log: {e}")

        # Add to failed scenarios in progress data
        if "failed_scenarios" not in self.progress_data:
            self.progress_data["failed_scenarios"] = []

        # Use scenario_id if available, otherwise use iteration
        failed_id = scenario_id if scenario_id else iteration
        self.progress_data["failed_scenarios"].append(failed_id)

        # Save progress immediately
        if hasattr(self, "progress_file") and self.progress_file:
            try:
                from src.collect_data.run_collection_utils import save_progress

                save_progress(self.progress_file, self.progress_data)
            except Exception as e:
                print(f"Error saving progress: {e}")

    def save_state(self):
        """Save current state to checkpoint and progress files."""
        if hasattr(self, "checkpoint_file") and self.checkpoint_file:
            save_checkpoint(self.checkpoint_file, self.processed_scenarios)
        if hasattr(self, "progress_file") and self.progress_file:
            save_progress(self.progress_file, self.progress_data)

    def print_final_summary(self, total_scenarios):
        """Print the final summary of the collection process."""
        print(f"\nData collection completed!")
        print(f"Successfully processed: {self.success_sum}/{total_scenarios} scenarios")

        if self.success_sum > 0:
            print(
                f"Average Spearman - Best: {self.spearman_sums['best']/self.success_sum:.3f}, "
                f"Worst: {self.spearman_sums['worst']/self.success_sum:.3f}, "
                f"Median: {self.spearman_sums['median']/self.success_sum:.3f}"
            )
            print(
                f"Average Cosine - Best: {self.cosine_sums['best']/self.success_sum:.3f}, "
                f"Worst: {self.cosine_sums['worst']/self.success_sum:.3f}, "
                f"Median: {self.cosine_sums['median']/self.success_sum:.3f}"
            )
            print(f"Average processing time per scenario: {self.total_time_sum/self.success_sum:.2f}s")

    def cleanup(self):
        """Clean up resources."""
        if self.use_wandb:
            wandb.finish()

    def run_collection(self, dataset):
        """Main collection loop - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run_collection")

    def _save_periodic_checkpoint(self, iteration):
        """Save checkpoint periodically (every 10 iterations)."""
        if iteration % 10 == 0:
            self.save_state()
