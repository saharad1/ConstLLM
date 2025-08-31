#!/usr/bin/env python3
"""
TruthfulQA Utilities and Benchmark Setup

This utility script provides helper functions for setting up and managing
TruthfulQA evaluations, including data preprocessing, result analysis,
and integration with the official TruthfulQA repository.

Features:
- Download and setup TruthfulQA benchmark data
- Result analysis and comparison tools
- Integration utilities for external judges
- Batch evaluation helpers
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import shutil
import urllib.request
from datetime import datetime

import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TruthfulQABenchmarkManager:
    """Manager for TruthfulQA benchmark setup and data handling."""
    
    def __init__(self, base_dir: str = "data/truthfulqa"):
        """
        Initialize benchmark manager.
        
        Args:
            base_dir: Base directory for TruthfulQA data and results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        self.external_dir = self.base_dir / "external"
        
        # Create subdirectories
        for dir_path in [self.data_dir, self.results_dir, self.external_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_benchmark(self, download_external: bool = False) -> bool:
        """
        Set up the TruthfulQA benchmark environment.
        
        Args:
            download_external: Whether to download external judge resources
            
        Returns:
            True if setup successful
        """
        logger.info("Setting up TruthfulQA benchmark...")
        
        try:
            # Download and cache datasets
            logger.info("Downloading TruthfulQA datasets...")
            mc_dataset = datasets.load_dataset("truthful_qa", "multiple_choice", cache_dir=str(self.data_dir))
            gen_dataset = datasets.load_dataset("truthful_qa", "generation", cache_dir=str(self.data_dir))
            
            # Save local copies for easy access
            mc_data = mc_dataset["validation"].to_pandas()
            gen_data = gen_dataset["validation"].to_pandas()
            
            mc_data.to_json(self.data_dir / "truthfulqa_mc.json", orient="records", indent=2)
            gen_data.to_json(self.data_dir / "truthfulqa_generation.json", orient="records", indent=2)
            
            logger.info(f"Saved {len(mc_data)} MC questions and {len(gen_data)} generation questions")
            
            # Download external resources if requested
            if download_external:
                self._download_external_resources()
            
            # Create configuration file
            self._create_config()
            
            logger.info("TruthfulQA benchmark setup complete!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up benchmark: {e}")
            return False
    
    def _download_external_resources(self):
        """Download external judge resources and original repository files."""
        logger.info("Downloading external TruthfulQA resources...")
        
        # URLs for key files from the original repository
        repo_files = {
            "TruthfulQA.csv": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
            "evaluate.py": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/evaluate.py"
        }
        
        for filename, url in repo_files.items():
            try:
                output_path = self.external_dir / filename
                if not output_path.exists():
                    logger.info(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, output_path)
                    logger.info(f"Downloaded {filename}")
                else:
                    logger.info(f"{filename} already exists")
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")
    
    def _create_config(self):
        """Create configuration file for benchmark settings."""
        config = {
            "benchmark_version": "1.0",
            "setup_date": datetime.now().isoformat(),
            "data_dir": str(self.data_dir),
            "results_dir": str(self.results_dir),
            "external_dir": str(self.external_dir),
            "datasets": {
                "multiple_choice": str(self.data_dir / "truthfulqa_mc.json"),
                "generation": str(self.data_dir / "truthfulqa_generation.json")
            }
        }
        
        config_path = self.base_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")


class TruthfulQAResultAnalyzer:
    """Analyzer for TruthfulQA evaluation results."""
    
    def __init__(self, results_dir: str = "results/truthfulqa"):
        """
        Initialize result analyzer.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
    
    def load_results(self, pattern: str = "*.json") -> Dict[str, Dict]:
        """
        Load all result files matching pattern.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            Dictionary mapping model names to results
        """
        results = {}
        
        for result_file in self.results_dir.glob(pattern):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    model_name = data.get("metrics", {}).get("model_name", result_file.stem)
                    results[model_name] = data
            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")
        
        return results
    
    def compare_models(self, result_files: List[str], output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Compare results across multiple models.
        
        Args:
            result_files: List of result file paths
            output_file: Optional output file for comparison table
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    metrics = data.get("metrics", {})
                    
                    row = {
                        "model": metrics.get("model_name", Path(file_path).stem),
                        "file": Path(file_path).name
                    }
                    
                    # Add MC metrics if available
                    if "mc1_accuracy" in metrics:
                        row["mc_accuracy"] = metrics["mc1_accuracy"]
                        row["mc_total"] = metrics.get("total_questions", 0)
                    
                    # Add generation metrics if available
                    if "truthfulness_percentage" in metrics:
                        row["truthfulness"] = metrics["truthfulness_percentage"]
                        row["informativeness"] = metrics["informativeness_percentage"]
                        row["both_truthful_informative"] = metrics.get("both_percentage", 0)
                        row["gen_total"] = metrics.get("total_questions", 0)
                    
                    # Add combined metrics if available
                    if "multiple_choice" in metrics:
                        row["mc_accuracy"] = metrics["multiple_choice"]["mc1_accuracy"]
                        row["mc_total"] = metrics["multiple_choice"]["total_questions"]
                    
                    if "generation" in metrics:
                        gen_metrics = metrics["generation"]
                        if "truthfulness_percentage" in gen_metrics:
                            row["truthfulness"] = gen_metrics["truthfulness_percentage"]
                            row["informativeness"] = gen_metrics["informativeness_percentage"]
                            row["both_truthful_informative"] = gen_metrics.get("both_percentage", 0)
                        row["gen_total"] = gen_metrics["total_questions"]
                    
                    comparison_data.append(row)
                    
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
        
        df = pd.DataFrame(comparison_data)
        
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Comparison saved to {output_file}")
        
        return df
    
    def create_summary_report(self, results_dir: Optional[str] = None, 
                            output_file: str = "truthfulqa_summary.md") -> str:
        """
        Create a markdown summary report of all results.
        
        Args:
            results_dir: Directory to analyze (uses self.results_dir if None)
            output_file: Output markdown file
            
        Returns:
            Path to created report
        """
        if results_dir:
            results_dir = Path(results_dir)
        else:
            results_dir = self.results_dir
        
        # Load all results
        results = {}
        for result_file in results_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    model_name = data.get("metrics", {}).get("model_name", result_file.stem)
                    results[model_name] = data
            except Exception as e:
                continue
        
        # Create markdown report
        report_lines = [
            "# TruthfulQA Evaluation Summary",
            f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTotal models evaluated: {len(results)}\n"
        ]
        
        if results:
            # Multiple Choice Results
            mc_results = []
            gen_results = []
            
            for model_name, data in results.items():
                metrics = data.get("metrics", {})
                
                # MC results
                if "mc1_accuracy" in metrics or "multiple_choice" in metrics:
                    mc_accuracy = metrics.get("mc1_accuracy")
                    if mc_accuracy is None and "multiple_choice" in metrics:
                        mc_accuracy = metrics["multiple_choice"]["mc1_accuracy"]
                    
                    if mc_accuracy is not None:
                        mc_results.append({
                            "model": model_name,
                            "accuracy": mc_accuracy,
                            "questions": metrics.get("total_questions", 
                                                   metrics.get("multiple_choice", {}).get("total_questions", 0))
                        })
                
                # Generation results
                gen_metrics = metrics
                if "generation" in metrics:
                    gen_metrics = metrics["generation"]
                
                if "truthfulness_percentage" in gen_metrics:
                    gen_results.append({
                        "model": model_name,
                        "truthfulness": gen_metrics["truthfulness_percentage"],
                        "informativeness": gen_metrics["informativeness_percentage"],
                        "both": gen_metrics.get("both_percentage", 0),
                        "questions": gen_metrics.get("total_questions", 0)
                    })
            
            # Add MC results table
            if mc_results:
                report_lines.extend([
                    "## Multiple Choice Results\n",
                    "| Model | Accuracy | Questions Evaluated |",
                    "|-------|----------|-------------------|"
                ])
                
                # Sort by accuracy
                mc_results.sort(key=lambda x: x["accuracy"], reverse=True)
                for result in mc_results:
                    report_lines.append(f"| {result['model']} | {result['accuracy']:.1f}% | {result['questions']} |")
            
            # Add generation results table
            if gen_results:
                report_lines.extend([
                    "\n## Generation Results\n",
                    "| Model | Truthfulness | Informativeness | Both | Questions |",
                    "|-------|--------------|-----------------|------|-----------|"
                ])
                
                # Sort by truthfulness
                gen_results.sort(key=lambda x: x["truthfulness"], reverse=True)
                for result in gen_results:
                    report_lines.append(
                        f"| {result['model']} | {result['truthfulness']:.1f}% | "
                        f"{result['informativeness']:.1f}% | {result['both']:.1f}% | {result['questions']} |"
                    )
            
            # Add best performing models
            if mc_results:
                best_mc = max(mc_results, key=lambda x: x["accuracy"])
                report_lines.extend([
                    "\n## Best Performing Models\n",
                    f"**Multiple Choice:** {best_mc['model']} ({best_mc['accuracy']:.1f}%)"
                ])
            
            if gen_results:
                best_truth = max(gen_results, key=lambda x: x["truthfulness"])
                best_both = max(gen_results, key=lambda x: x["both"])
                report_lines.extend([
                    f"**Generation (Truthfulness):** {best_truth['model']} ({best_truth['truthfulness']:.1f}%)",
                    f"**Generation (Both):** {best_both['model']} ({best_both['both']:.1f}%)"
                ])
        
        # Write report
        report_path = results_dir / output_file
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report created: {report_path}")
        return str(report_path)
    
    def plot_comparison(self, results: Dict[str, Dict], metric: str = "mc1_accuracy",
                       output_file: Optional[str] = None) -> str:
        """
        Create comparison plot for specified metric.
        
        Args:
            results: Results dictionary from load_results()
            metric: Metric to plot
            output_file: Output file for plot
            
        Returns:
            Path to created plot
        """
        # Extract data for plotting
        models = []
        values = []
        
        for model_name, data in results.items():
            metrics = data.get("metrics", {})
            value = None
            
            if metric in metrics:
                value = metrics[metric]
            elif "multiple_choice" in metrics and metric in metrics["multiple_choice"]:
                value = metrics["multiple_choice"][metric]
            elif "generation" in metrics and metric in metrics["generation"]:
                value = metrics["generation"][metric]
            
            if value is not None:
                models.append(model_name)
                values.append(value)
        
        if not models:
            logger.warning(f"No data found for metric: {metric}")
            return ""
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(models)), values)
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f"TruthfulQA {metric.replace('_', ' ').title()} Comparison")
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if not output_file:
            output_file = f"truthfulqa_{metric}_comparison.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved: {output_file}")
        return output_file


def run_batch_evaluation(model_list: List[str], script_name: str = "truthfulqa_comprehensive_eval.py",
                        output_dir: str = "results/truthfulqa_batch", **kwargs) -> List[str]:
    """
    Run batch evaluation on multiple models.
    
    Args:
        model_list: List of model names/paths to evaluate
        script_name: Evaluation script to use
        output_dir: Output directory for results
        **kwargs: Additional arguments for the evaluation script
        
    Returns:
        List of result file paths
    """
    logger.info(f"Running batch evaluation on {len(model_list)} models...")
    
    os.makedirs(output_dir, exist_ok=True)
    result_files = []
    
    for i, model_name in enumerate(model_list, 1):
        logger.info(f"Evaluating model {i}/{len(model_list)}: {model_name}")
        
        # Build command
        cmd = [
            "python", f"src/{script_name}",
            "--model_name", model_name,
            "--output_dir", output_dir
        ]
        
        # Add additional arguments
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])
        
        try:
            # Run evaluation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info(f"Successfully evaluated {model_name}")
                # Find result file (approximate - based on naming convention)
                model_safe_name = Path(model_name).name.replace("/", "_")
                result_pattern = f"*{model_safe_name}*.json"
                result_files.extend(list(Path(output_dir).glob(result_pattern)))
            else:
                logger.error(f"Failed to evaluate {model_name}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout evaluating {model_name}")
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
    
    logger.info(f"Batch evaluation complete. Results in {output_dir}")
    return result_files


def main():
    parser = argparse.ArgumentParser(description="TruthfulQA Utilities and Benchmark Setup")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup TruthfulQA benchmark")
    setup_parser.add_argument("--base_dir", type=str, default="data/truthfulqa",
                             help="Base directory for benchmark data")
    setup_parser.add_argument("--download_external", action="store_true",
                             help="Download external judge resources")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze evaluation results")
    analyze_parser.add_argument("--results_dir", type=str, default="results/truthfulqa",
                               help="Results directory to analyze")
    analyze_parser.add_argument("--output", type=str, default="truthfulqa_summary.md",
                               help="Output summary file")
    analyze_parser.add_argument("--plot", type=str, choices=["mc1_accuracy", "truthfulness_percentage"],
                               help="Create comparison plot for metric")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare specific result files")
    compare_parser.add_argument("files", nargs="+", help="Result files to compare")
    compare_parser.add_argument("--output", type=str, default="comparison.csv",
                               help="Output comparison file")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch evaluation")
    batch_parser.add_argument("--models", nargs="+", required=True,
                             help="List of models to evaluate")
    batch_parser.add_argument("--script", type=str, default="truthfulqa_comprehensive_eval.py",
                             help="Evaluation script to use")
    batch_parser.add_argument("--output_dir", type=str, default="results/truthfulqa_batch",
                             help="Output directory")
    batch_parser.add_argument("--task", type=str, choices=["mc", "generation", "both"],
                             default="both", help="Task to run")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        manager = TruthfulQABenchmarkManager(args.base_dir)
        success = manager.setup_benchmark(download_external=args.download_external)
        if success:
            print(f"TruthfulQA benchmark setup complete in {args.base_dir}")
        else:
            print("Setup failed!")
            sys.exit(1)
    
    elif args.command == "analyze":
        analyzer = TruthfulQAResultAnalyzer(args.results_dir)
        report_path = analyzer.create_summary_report(output_file=args.output)
        print(f"Summary report created: {report_path}")
        
        if args.plot:
            results = analyzer.load_results()
            plot_path = analyzer.plot_comparison(results, metric=args.plot)
            if plot_path:
                print(f"Comparison plot created: {plot_path}")
    
    elif args.command == "compare":
        analyzer = TruthfulQAResultAnalyzer()
        df = analyzer.compare_models(args.files, output_file=args.output)
        print(f"Comparison saved to {args.output}")
        print("\nComparison preview:")
        print(df.to_string())
    
    elif args.command == "batch":
        result_files = run_batch_evaluation(
            model_list=args.models,
            script_name=args.script,
            output_dir=args.output_dir,
            task=args.task
        )
        print(f"Batch evaluation complete. {len(result_files)} result files created.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()