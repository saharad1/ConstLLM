#!/bin/bash

# Script to resume an evaluation run by providing the folder path
# Usage: ./resume_eval_run.sh <run_folder_path> [additional_options]

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0

# Default parameters - adjust these as needed
# Set RUN_FOLDER_PATH here to avoid typing it every time
RUN_FOLDER_PATH="data/eval_results/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_250806_181728_lr6.86e-06_beta8.41/eval_250829_172353_test_521_LIG_no_pregen"

NUM_DEC_EXP=5
TEMPERATURE=0.7
SEED=42
USE_WANDB=true
IGNORE_PRE_GENERATED=true
OUTPUT_DIR=""
SUBSET=""

# Function to display usage
usage() {
    echo "Usage: $0 [run_folder_path] [options]"
    echo ""
    echo "Arguments:"
    echo "  run_folder_path    Path to the evaluation run folder to resume (can also be set at top of script)"
    echo ""
    echo "Options:"
    echo "  --run_folder_path|-r  Set the run folder path"
    echo "  --attribution_method|-a  Override attribution method (default: extracted from run name)"
    echo "  --num_dec_exp|-n      Override number of decision explanations (default: $NUM_DEC_EXP)"
    echo "  --temperature|-t      Override temperature (default: $TEMPERATURE)"
    echo "  --seed                Override seed (default: $SEED)"
    echo "  --wandb|-w            Enable wandb logging (default: $USE_WANDB)"
    echo "  --no_wandb            Disable wandb logging"
    echo "  --ignore_pre_generated  Ignore pre-generated responses"
    echo "  --output_dir|-o       Override output directory"
    echo "  --subset|-s           Override subset size"
    echo "  --help|-h             Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Set RUN_FOLDER_PATH at top of script, then run:"
    echo "  $0"
    echo ""
    echo "  # Provide run folder path as argument:"
    echo "  $0 data/eval_results/arc_easy/huggingface/Meta-Llama-3.1-8B-Instruct/eval_250824_172702_test_521_LIG_with_pregen"
    echo ""
    echo "  # Override parameters:"
    echo "  $0 --run_folder_path data/eval_results/arc_easy/huggingface/Meta-Llama-3.1-8B-Instruct/eval_250824_172702_test_521_LIG_with_pregen --temperature 0.5"
    echo ""
    echo "Note: The script automatically loads the original configuration from progress.json"
    echo "      including model_path, dataset_path, and other parameters from the original run."
    echo "      You can set RUN_FOLDER_PATH at the top of this script for convenience."
}

# Initialize array for remaining arguments
REMAINING_ARGS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h)
      usage
      exit 0
      ;;
    --run_folder_path|-r)
      RUN_FOLDER_PATH="$2"
      shift 2
      ;;
    --num_dec_exp|-n)
      NUM_DEC_EXP="$2"
      shift 2
      ;;
    --temperature|-t)
      TEMPERATURE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --wandb|-w)
      USE_WANDB=true
      shift
      ;;
    --no_wandb)
      USE_WANDB=false
      shift
      ;;
    --ignore_pre_generated)
      IGNORE_PRE_GENERATED=true
      shift
      ;;
    --output_dir|-o)
      OUTPUT_DIR="--output_dir $2"
      shift 2
      ;;
    --subset|-s)
      SUBSET="--subset $2"
      shift 2
      ;;
    --attribution_method|-a)
      ATTRIBUTION_METHOD="$2"
      shift 2
      ;;
    *)
      # If no flag is provided and RUN_FOLDER_PATH is empty, treat as run folder path
      if [ -z "$RUN_FOLDER_PATH" ]; then
        RUN_FOLDER_PATH="$1"
      else
        # Pass remaining arguments to eval_trained_dpo.sh, but filter out --help
        if [ "$1" != "--help" ] && [ "$1" != "-h" ]; then
          REMAINING_ARGS+=("$1")
        fi
      fi
      shift
      ;;
  esac
done

# Check if run folder path is provided
if [ -z "$RUN_FOLDER_PATH" ]; then
    echo "Error: Run folder path is required"
    echo "You can provide it as:"
    echo "  1. First argument: $0 <run_folder_path>"
    echo "  2. Using flag: $0 --run_folder_path <run_folder_path>"
    echo "  3. Set RUN_FOLDER_PATH at the top of this script"
    echo ""
    usage
    exit 1
fi

# Check if the run folder exists
if [ ! -d "$RUN_FOLDER_PATH" ]; then
    echo "Error: Run folder does not exist: $RUN_FOLDER_PATH"
    exit 1
fi

# Check if progress.json exists in the run folder
PROGRESS_FILE="$RUN_FOLDER_PATH/progress.json"
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "Error: progress.json not found in run folder: $RUN_FOLDER_PATH"
    echo "This file is required to resume the run with the original configuration."
    exit 1
fi

# Extract run name from the folder path
RUN_NAME=$(basename "$RUN_FOLDER_PATH")
echo "Resuming run: $RUN_NAME"

# Load original configuration from progress.json
echo "Loading original configuration from progress.json..."
ORIGINAL_MODEL_PATH=$(python3 -c "import json; data=json.load(open('$PROGRESS_FILE')); print(data.get('model_path', ''))")
ORIGINAL_DATASET_PATH=$(python3 -c "import json; data=json.load(open('$PROGRESS_FILE')); print(data.get('dataset_path', ''))")

if [ -z "$ORIGINAL_MODEL_PATH" ] || [ -z "$ORIGINAL_DATASET_PATH" ]; then
    echo "Error: Could not load original model_path or dataset_path from progress.json"
    exit 1
fi

# Extract attribution method and pregen setting from run name
# Run name format: eval_<timestamp>_<dataset_name>_<attribution_method>_<pregen_signal>
# Example: eval_250824_172702_test_521_LIG_with_pregen
ATTRIBUTION_METHOD=$(echo "$RUN_NAME" | sed 's/.*_\([^_]*\)_\(with_pregen\|no_pregen\)$/\1/')

# Extract pregen setting from run name
if [[ "$RUN_NAME" == *"no_pregen" ]]; then
    IGNORE_PRE_GENERATED=true
    echo "  Pre-generated responses: IGNORED (extracted from run name)"
else
    IGNORE_PRE_GENERATED=false
    echo "  Pre-generated responses: USED (extracted from run name)"
fi

echo "Original configuration:"
echo "  Model path: $ORIGINAL_MODEL_PATH"
echo "  Dataset path: $ORIGINAL_DATASET_PATH"
echo "  Attribution method: $ATTRIBUTION_METHOD (extracted from run name)"

# Determine if this is a HuggingFace model ID or local path
if [[ "$ORIGINAL_MODEL_PATH" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$ ]] && [[ ! "$ORIGINAL_MODEL_PATH" =~ ^(\.|/|models/) ]]; then
    IS_MODEL_ID="--is_model_id"
    echo "  Model type: HuggingFace model ID"
else
    IS_MODEL_ID=""
    echo "  Model type: Local model path"
fi

echo ""
echo "Calling eval_trained_dpo.sh with original configuration..."

# Call the eval_trained_dpo.sh script with the original configuration
bash scripts/eval_trained_dpo.sh \
  --model_path "$ORIGINAL_MODEL_PATH" \
  --dataset_path "$ORIGINAL_DATASET_PATH" \
  --attribution_method "$ATTRIBUTION_METHOD" \
  --num_dec_exp "$NUM_DEC_EXP" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --resume_run "$RUN_NAME" \
  $([ "$USE_WANDB" = true ] && echo "--wandb") \
  $([ "$IGNORE_PRE_GENERATED" = true ] && echo "--ignore_pre_generated") \
  $([ -n "$OUTPUT_DIR" ] && echo "$OUTPUT_DIR") \
  $([ -n "$SUBSET" ] && echo "$SUBSET") \
  $([ -n "$IS_MODEL_ID" ] && echo "$IS_MODEL_ID") \
  "${REMAINING_ARGS[@]}"

echo "Resume evaluation completed!"
