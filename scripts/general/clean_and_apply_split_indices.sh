#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Default values
INPUT_FILE="data/collection_data/ecqa/meta-llama_Meta-Llama-3.1-8B-Instruct/ecqa_20250521_120325_LIG_llama3.1/ecqa_20250521_120325_LIG_llama3.1.jsonl"
SPLIT_FILE="data/dataset_splits/ecqa_split_indices.json"
KEEP_CLEANED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --split_file|-s)
            SPLIT_FILE="$2"
            shift 2
            ;;
        --keep_cleaned|-k)
            KEEP_CLEANED=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --input_file <path> --split_file <path> [options]"
            echo "Options:"
            echo "  --keep_cleaned            Keep the intermediate cleaned file"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: --input_file is required"
    exit 1
fi

if [[ -z "$SPLIT_FILE" ]]; then
    echo "Error: --split_file is required"
    exit 1
fi

# Build the command
CMD="python -m src.prepare_datasets.clean_and_apply_split_indices \"$INPUT_FILE\" \"$SPLIT_FILE\""

if [[ "$KEEP_CLEANED" == true ]]; then
    CMD="$CMD --keep_cleaned"
fi

# Execute the command
echo "Running: $CMD"
eval $CMD 