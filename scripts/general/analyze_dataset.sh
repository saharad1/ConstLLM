#!/bin/bash
#
# Dataset Analysis Script for ConstLLM
# This script analyzes generated datasets and computes various metrics
#

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Exit on error
set -e

# data/eval_results/arc_easy/Llama-3.2-3B-Instruct/arc_easy_250805_195416_lr3.84e-06_beta9.20/eval_250817_134151_test_521_LIG_with_pregen/eval_250817_134151_test_521_LIG_with_pregen_results.jsonl
# data/eval_results/arc_easy/huggingface/Llama-3.2-3B-Instruct/eval_250818_090347_test_521_LIG_with_pregen/eval_250818_090347_test_521_LIG_with_pregen_results.jsonl
# Default values
DATASET_PATH="data/collection_data/ecqa/meta-llama_Meta-Llama-3.1-8B-Instruct/ecqa_20250521_120325_LIG_llama3.1/ecqa_20250521_120325_LIG_llama3.1_dedup.jsonl"

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -f, --file PATH            Path to the dataset file (required)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -f data/collection_data/ecqa/test_1090.jsonl"
    echo "  $0 -f data/collection_data/codah/dataset.jsonl"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            DATASET_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check if dataset path is provided
if [[ -z "$DATASET_PATH" ]]; then
    echo "Error: Dataset file path is required."
    echo "Use -f or --file to specify the dataset file."
    show_help
fi

# Check if file exists
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Error: Dataset file '$DATASET_PATH' does not exist."
    exit 1
fi

# Check if we're in the correct directory (should have src/ directory)
if [[ ! -d "src" ]]; then
    echo "Error: Please run this script from the project root directory (where src/ directory is located)."
    echo "Current directory: $(pwd)"
    exit 1
fi

# Activate environment if not already activated
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "ConstLLM" ]]; then
    echo "Activating ConstLLM conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ConstLLM || { echo "Failed to activate ConstLLM environment. Exiting."; exit 1; }
fi

# Create temporary Python script for analysis
TEMP_SCRIPT="/tmp/temp_analysis_$$.py"
cat > "$TEMP_SCRIPT" << 'EOF'
import sys
import os

# Get the project root directory (assuming script is run from project root)
project_root = os.getcwd()
sys.path.insert(0, project_root)

from src.analyze_data.analyze_generated_dataset import analyze_dataset, print_metrics

# Analyze dataset
print('Analyzing dataset: DATASET_PATH_PLACEHOLDER')
metrics = analyze_dataset('DATASET_PATH_PLACEHOLDER')
print_metrics(metrics)
EOF

# Replace placeholder with actual dataset path
sed -i "s|DATASET_PATH_PLACEHOLDER|$DATASET_PATH|g" "$TEMP_SCRIPT"

# Run the analysis
echo "Running dataset analysis..."
python "$TEMP_SCRIPT"

# Clean up temporary file
rm -f "$TEMP_SCRIPT"

echo "Analysis completed!" 