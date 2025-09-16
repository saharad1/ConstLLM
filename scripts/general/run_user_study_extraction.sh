#!/bin/bash

# User Study Sample Extraction Script
# Usage: ./run_user_study_extraction.sh --datasets "dataset1,dataset2,..." --num-samples N
# Or: ./run_user_study_extraction.sh --auto-discover

set -e

# Default values
DEFAULT_NUM_SAMPLES=5
OUTPUT_DIR="outputs/user_study_extraction"
DEFAULT_OUTPUT_FILE="$OUTPUT_DIR/user_study_samples.md"
DEFAULT_CSV_FILE="$OUTPUT_DIR/user_study_samples.csv"
PYTHON_SCRIPT="src/extract_user_study_samples.py"

# Help function
show_help() {
    cat << EOF
User Study Sample Extraction Script

Usage:
    $0 [OPTIONS]

OPTIONS:
    -d, --datasets DATASETS     Comma-separated list of dataset names (e.g., "arc_challenge,hellaswag")
    -f, --files FILES           Comma-separated list of specific jsonl file paths
    -n, --num-samples N         Number of samples per dataset/model combination (default: $DEFAULT_NUM_SAMPLES)
    -o, --output OUTPUT_FILE    Markdown output file path (default: $DEFAULT_OUTPUT_FILE)
    -c, --csv CSV_FILE          CSV output file path (default: $DEFAULT_CSV_FILE)
    -a, --auto-discover         Auto-discover all evaluation files in standard locations
    -h, --help                  Show this help message

EXAMPLES:
    # Extract 3 samples from specific datasets
    $0 --datasets "arc_challenge,hellaswag,mmlu" --num-samples 3

    # Extract samples from specific evaluation files
    $0 --files "file1.jsonl,file2.jsonl,file3.jsonl" --num-samples 5

    # Auto-discover all evaluation files and extract 5 samples each
    $0 --auto-discover --num-samples 5

    # Specify custom output files
    $0 --datasets "arc_challenge" --output "my_study.md" --csv "my_study.csv"

EVALUATION FILE DISCOVERY:
    The script will automatically find evaluation files for the specified datasets.
    Expected directory structure:
    - /raid/saharad/ConstLLM/data/collection_data/DATASET/MODEL/.../*.jsonl
    - /raid/saharad/ConstLLM/data/eval_results/DATASET/MODEL/.../*results.jsonl

EOF
}

# Parse command line arguments
NUM_SAMPLES=$DEFAULT_NUM_SAMPLES
OUTPUT_FILE=$DEFAULT_OUTPUT_FILE
CSV_FILE=$DEFAULT_CSV_FILE
AUTO_DISCOVER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--datasets)
            DATASETS="$2"
            shift 2
            ;;
        -f|--files)
            FILES="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -c|--csv)
            CSV_FILE="$2"
            shift 2
            ;;
        -a|--auto-discover)
            AUTO_DISCOVER=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validation
if ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]] || [ "$NUM_SAMPLES" -le 0 ]; then
    echo "Error: num-samples must be a positive integer, got: $NUM_SAMPLES"
    exit 1
fi

if [[ -z "$DATASETS" ]] && [[ -z "$FILES" ]] && [[ "$AUTO_DISCOVER" != true ]]; then
    echo "Error: No datasets or files specified. Use --datasets, --files, or --auto-discover"
    show_help
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Display configuration
echo "=========================================="
echo "User Study Sample Extraction Configuration"
echo "=========================================="
if [[ "$AUTO_DISCOVER" == true ]]; then
    echo "Datasets: Auto-discovering all evaluation datasets"
elif [[ -n "$FILES" ]]; then
    echo "Files: $FILES"
else
    echo "Datasets: $DATASETS"
fi
echo "Samples per combination: $NUM_SAMPLES"
echo "Markdown output file: $OUTPUT_FILE"
echo "CSV output file: $CSV_FILE"
echo "=========================================="

# Build command line arguments for the Python script
PYTHON_ARGS=()

if [[ -n "$DATASETS" ]]; then
    PYTHON_ARGS+=("--datasets" "$DATASETS")
fi

if [[ -n "$FILES" ]]; then
    PYTHON_ARGS+=("--files" "$FILES")
fi

if [[ "$AUTO_DISCOVER" == true ]]; then
    PYTHON_ARGS+=("--auto-discover")
fi

PYTHON_ARGS+=("--num-samples" "$NUM_SAMPLES")
PYTHON_ARGS+=("--output" "$OUTPUT_FILE")
PYTHON_ARGS+=("--csv" "$CSV_FILE")

# Run the Python script
echo ""
echo "Running extraction..."
python3 "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"

echo ""
echo "✅ User study extraction completed!"
echo "📁 Markdown file: $OUTPUT_FILE"
echo "📄 CSV file: $CSV_FILE"
