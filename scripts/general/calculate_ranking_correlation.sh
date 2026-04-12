#!/bin/bash

# Script to calculate explanation ranking correlation between two datasets
# Usage: ./calculate_ranking_correlation.sh --dataset1_path <path> --dataset2_path <path> [--metric_type <type>] [--output_dir <dir>]

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

set -e  # Exit on any error

# Default values
DATASET1_PATH="data/collection_data/ecqa/meta-llama_Meta-Llama-3.1-8B-Instruct/ecqa_20250521_120325_LIG_llama3.1/ecqa_20250521_120325_LIG_llama3.1_dedup.jsonl"
DATASET2_PATH="data/collection_data/ecqa/unsloth_Meta-Llama-3.1-8B-Instruct/ecqa_20250404_120218_LIME_llama3.1/ecqa_20250404_120218_LIME_llama3.1_fixed.jsonl"
RANKING_METRIC="spearman"
OUTPUT_DIR="results/correlation_results"

# Function to display usage
usage() {
    echo "Usage: $0 [--dataset1_path <path>] [--dataset2_path <path>] [--ranking_metric <type>] [--output_dir <dir>]"
    echo ""
    echo "Arguments (all optional - will use defaults if not provided):"
    echo "  dataset1_path  : Path to the first dataset (JSONL file)"
    echo "  dataset2_path  : Path to the second dataset (JSONL file)"
    echo "  ranking_metric : Metric to rank explanations within scenarios (default: spearman)"
    echo "                  Options: spearman, cosine, lma"
    echo "  output_dir     : Directory to save results (default: results/correlation_results)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default datasets"
    echo "  $0 --dataset1_path dataset1.jsonl --dataset2_path dataset2.jsonl"
    echo "  $0 --dataset1_path dataset1.jsonl --dataset2_path dataset2.jsonl --ranking_metric cosine"
    echo "  $0 --dataset1_path dataset1.jsonl --dataset2_path dataset2.jsonl --ranking_metric spearman --output_dir my_results"
    echo ""
    echo "Output files:"
    echo "  - correlation_analysis.txt: Detailed text results"
    echo "  - correlation_plot.png: Visualization plot"
    echo "  - correlation_summary.json: JSON summary of results"
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Check if no arguments provided - use defaults
if [[ $# -eq 0 ]]; then
    echo "No arguments provided, using default datasets:"
    echo "  Dataset 1: $DATASET1_PATH"
    echo "  Dataset 2: $DATASET2_PATH"
    echo "  Ranking Metric: $RANKING_METRIC"
    echo "  Output: $OUTPUT_DIR"
    echo ""
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset1_path)
            DATASET1_PATH="$2"
            shift 2
            ;;
        --dataset2_path)
            DATASET2_PATH="$2"
            shift 2
            ;;
        --ranking_metric)
            RANKING_METRIC="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate that we have dataset paths (either from arguments or defaults)
if [[ -z "$DATASET1_PATH" ]]; then
    echo "Error: No dataset1_path provided and no default set"
    usage
    exit 1
fi

if [[ -z "$DATASET2_PATH" ]]; then
    echo "Error: No dataset2_path provided and no default set"
    usage
    exit 1
fi

# Validate ranking metric
if [[ ! "$RANKING_METRIC" =~ ^(spearman|cosine|lma)$ ]]; then
    echo "Error: Invalid ranking metric '$RANKING_METRIC'. Must be one of: spearman, cosine, lma"
    exit 1
fi

# Check if datasets exist
if [[ ! -f "$DATASET1_PATH" ]]; then
    echo "Error: Dataset 1 not found: $DATASET1_PATH"
    exit 1
fi

if [[ ! -f "$DATASET2_PATH" ]]; then
    echo "Error: Dataset 2 not found: $DATASET2_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PREFIX="${OUTPUT_DIR}/correlation_${RANKING_METRIC}_${TIMESTAMP}"

echo "=========================================="
echo "Explanation Ranking Correlation Analysis"
echo "=========================================="
echo "Dataset 1: $DATASET1_PATH"
echo "Dataset 2: $DATASET2_PATH"
echo "Ranking Metric: $RANKING_METRIC"
echo "Output Directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Use the standalone Python script
PYTHON_SCRIPT="/raid/saharad/ConstLLM/src/analyze_data/run_correlation_analysis.py"

# Run the analysis
echo "Running correlation analysis..."
cd /raid/saharad/ConstLLM
export PYTHONPATH="/raid/saharad/ConstLLM:$PYTHONPATH"
python "$PYTHON_SCRIPT" "$DATASET1_PATH" "$DATASET2_PATH" "$RANKING_METRIC" "$OUTPUT_PREFIX"

echo ""
echo "Analysis completed successfully!"
echo "Check the output directory '$OUTPUT_DIR' for results."
