#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Default values
INPUT_FILE="data/collection_data/arc_easy/unsloth_Qwen2.5-7B-Instruct/arc_easy_20250417_125849_LIME_Qwen2.5/arc_easy_20250417_125849_LIME_Qwen2.5.jsonl"
TRAIN_RATIO=0.7
EVAL_RATIO=0.2
TEST_RATIO=0.1
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --train_ratio|-t)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --eval_ratio|-e)
            EVAL_RATIO="$2"
            shift 2
            ;;
        --test_ratio|-r)
            TEST_RATIO="$2"
            shift 2
            ;;
        --seed|-s)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--input_file <path_to_file>] [--output_dir <output_dir>] [--train_ratio <ratio>] [--eval_ratio <ratio>] [--test_ratio <ratio>] [--seed <seed>]"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python -m src.pipeline_dpo.clean_split_dataset \"$INPUT_FILE\""

# Add optional arguments if provided
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

CMD="$CMD --train_ratio $TRAIN_RATIO --eval_ratio $EVAL_RATIO --test_ratio $TEST_RATIO --seed $SEED"

# Execute the command
echo "Running: $CMD"
eval $CMD
