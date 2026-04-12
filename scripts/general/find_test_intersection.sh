#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Default values
TEST_FILES=(
  "data/collection_data/arc_easy/meta-llama_Llama-3.2-3B-Instruct/arc_easy_20250404_115258_LIG_llama3.2/test_521.jsonl"
  "data/collection_data/arc_easy/meta-llama_Meta-Llama-3.1-8B-Instruct/arc_easy_20250527_100818_LIG_llama3.1/test_521.jsonl"
  "data/collection_data/arc_easy/unsloth_Llama-3.2-3B-Instruct/arc_easy_20250417_124056_LIME_llama3.2/test_517.jsonl"
  "data/collection_data/arc_easy/unsloth_Meta-Llama-3.1-8B-Instruct/arc_easy_20250424_152104_LIME_llama3.1/test_521.jsonl"
)
SEARCH_DIR=""
DATASET_NAME="arc_easy"
OUTPUT_DIR="data/intersection_test_sets"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test_files)
            shift  # Remove --test_files
            TEST_FILES=()
            # Collect all remaining arguments as test files until we hit another option
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                TEST_FILES+=("$1")
                shift
            done
            ;;
        --search_dir)
            SEARCH_DIR="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test_files <file1> <file2> ...] [--search_dir <dir>] [--dataset_name <name>] [--output_dir <dir>]"
            echo ""
            echo "Options:"
            echo "  --test_files <files>     Explicit list of test files"
            echo "  --search_dir <dir>       Directory to search for test files"
            echo "  --dataset_name <name>    Dataset name for filtering and output"
            echo "  --output_dir <dir>       Output directory (default: data/consistent_test_sets)"
            echo ""
            echo "Examples:"
            echo "  $0 --test_files data/test1.jsonl data/test2.jsonl --dataset_name ecqa"
            echo "  $0 --search_dir data/collection_data/ecqa --dataset_name ecqa"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python -m src.find_test_set_intersection"

if [[ ${#TEST_FILES[@]} -gt 0 ]]; then
    CMD="$CMD --test_files"
    for file in "${TEST_FILES[@]}"; do
        if [[ -n "$file" ]]; then  # Skip empty strings
            CMD="$CMD $file"
        fi
    done
fi

if [[ -n "$SEARCH_DIR" ]]; then
    CMD="$CMD --search_dir $SEARCH_DIR"
fi

if [[ -n "$DATASET_NAME" ]]; then
    CMD="$CMD --dataset_name $DATASET_NAME"
fi

CMD="$CMD --output_dir $OUTPUT_DIR"

# Execute the command
echo "Running: $CMD"
eval $CMD 