#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Default values
INPUT_FILE=""
SPLIT_FILE=""
OUTPUT_DIR=""

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
        --output_dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --input_file <path> --split_file <path> [options]"
            echo "Options:"
            echo "  --output_dir <dir>        Output directory (default: same as input file)"
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
CMD="python -m src.pipeline_dpo.apply_split_indices \"$INPUT_FILE\" \"$SPLIT_FILE\""

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

# Execute the command
echo "Running: $CMD"
eval $CMD 