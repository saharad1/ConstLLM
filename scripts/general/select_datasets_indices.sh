#!/bin/bash
#
# Script to select random indices from dataset test sets
# This script randomly selects 250 indices from each dataset's test set
#

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Exit on error
set -e

# Default values
SPLITS_DIR="data/dataset_splits"
OUTPUT_FILE="data/dataset_splits/datasets_test_indices.json"
NUM_SAMPLES=250
SEED=42

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --splits_dir DIR          Directory containing dataset split indices files (default: data/dataset_splits)"
    echo "  --output_file FILE        Output JSON file path (default: data/dataset_splits/datasets_test_indices.json)"
    echo "  --num_samples N           Number of indices to select from each test set (default: 250)"
    echo "  --seed SEED               Random seed for reproducibility (default: 42)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all defaults"
    echo "  $0 --num_samples 500                  # Select 500 indices instead of 250"
    echo "  $0 --seed 123 --output_file custom.json  # Use custom seed and output file"
    echo ""
    echo "This script will:"
    echo "  1. Load test indices from each dataset's split indices file"
    echo "  2. Randomly select the specified number of indices from each test set"
    echo "  3. Create a JSON file with the selected indices organized by dataset"
    echo ""
    echo "Supported datasets: arc_easy, arc_challenge, ecqa, codah"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --splits_dir)
            SPLITS_DIR="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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

# Check if we're in the correct directory (should have src/ directory)
if [[ ! -d "src" ]]; then
    echo "Error: Please run this script from the project root directory (where src/ directory is located)."
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if splits directory exists
if [[ ! -d "$SPLITS_DIR" ]]; then
    echo "Error: Splits directory '$SPLITS_DIR' does not exist."
    echo "Please run the create_split_indices.sh script first to generate the split indices files."
    exit 1
fi

# Activate environment if not already activated
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "ConstLLM" ]]; then
    echo "Activating ConstLLM conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ConstLLM || { echo "Failed to activate ConstLLM environment. Exiting."; exit 1; }
fi

echo "Selecting dataset indices..."
echo "Splits directory: $SPLITS_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Number of samples per dataset: $NUM_SAMPLES"
echo "Random seed: $SEED"
echo ""

# Run the Python script
python src/select_dataset_indices.py \
    --splits_dir "$SPLITS_DIR" \
    --output_file "$OUTPUT_FILE" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED"

echo ""
echo "Dataset indices selection completed!"
echo "You can now use the generated indices file for your data collection."
