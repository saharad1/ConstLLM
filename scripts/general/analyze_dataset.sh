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

# Default values
DATASET_PATH="data/collection_data/arc_easy/meta-llama_Meta-Llama-3.1-8B-Instruct/arc_easy_20250527_100818_LIG_llama3.1/test_521.jsonl"

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
sys.path.append('src')
from analyze_data.analyze_generated_dataset import analyze_dataset, print_metrics

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