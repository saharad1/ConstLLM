#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Default values
# To this:
TEST_FILES=(
  "data/collection_data/ecqa/meta-llama_Llama-3.2-3B-Instruct/ecqa_20250403_133655_LIG_llama3.2/test_1090.jsonl"
  "data/collection_data/ecqa/unsloth_Llama-3.2-3B-Instruct/ecqa_20250506_085629_LIME_llama3.2/test_1085.jsonl"
  "data/collection_data/ecqa/unsloth_Meta-Llama-3.1-8B-Instruct/ecqa_20250508_173142_LIME_llama3.1/test_1088.jsonl"
)
SEARCH_DIR=""
DATASET_NAME=""
OUTPUT_DIR="data/consistent_test_sets"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test_files)
            TEST_FILES="$2"
            shift 2
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

if [[ -n "$TEST_FILES" ]]; then
    CMD="$CMD --test_files $TEST_FILES"
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