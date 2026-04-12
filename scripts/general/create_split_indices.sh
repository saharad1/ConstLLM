#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Datasets names:
# ecqa
# arc_easy
# arc_challenge
# codah

# Default values
DATASET_NAME="codah"
SUBSET=""
TRAIN_RATIO=0.7
EVAL_RATIO=0.2
TEST_RATIO=0.1
SEED=42
OUTPUT_DIR="data/dataset_splits"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|-d)
            DATASET_NAME="$2"
            shift 2
            ;;
        --subset|-s)
            SUBSET="$2"
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
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --dataset <dataset_name> [options]"
            echo "Options:"
            echo "  --subset <size>           Subset size"
            echo "  --train_ratio <ratio>     Training ratio (default: 0.7)"
            echo "  --eval_ratio <ratio>      Evaluation ratio (default: 0.2)"
            echo "  --test_ratio <ratio>      Test ratio (default: 0.1)"
            echo "  --seed <seed>             Random seed (default: 42)"
            echo "  --output_dir <dir>        Output directory (default: data/dataset_splits)"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: --dataset is required"
    exit 1
fi

# Build the command
CMD="python -m src.prepare_datasets.create_split_indices $DATASET_NAME"

# Add optional arguments
if [[ -n "$SUBSET" ]]; then
    CMD="$CMD --subset $SUBSET"
fi

CMD="$CMD --train_ratio $TRAIN_RATIO --eval_ratio $EVAL_RATIO --test_ratio $TEST_RATIO --seed $SEED --output_dir $OUTPUT_DIR"

# Execute the command
echo "Running: $CMD"
eval $CMD 