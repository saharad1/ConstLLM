#!/bin/bash
#
# Kernel SHAP Data Collection Script for ConstLLM
# This script runs the data collection process specifically for kernel SHAP analysis
# using pre-selected scenario IDs from the test sets
#

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Exit on error
set -e

# GPU ID to use
export CUDA_VISIBLE_DEVICES=2

# Model ids:
# unsloth/mistral-7b-instruct-v0.3
# unsloth/Meta-Llama-3.1-8B-Instruct
# unsloth/Qwen2.5-7B-Instruct
# unsloth/Llama-3.2-3B-Instruct
# For LIG:
# meta-llama/Meta-Llama-3.1-8B-Instruct
# mistralai/Mistral-7B-Instruct-v0.3
# meta-llama/Llama-3.2-3B-Instruct

# Attribution methods:
# LIG
# LIME
# Feature Ablation

# Datasets:
# ecqa
# choice75
# codah
# arc_easy
# arc_challenge

# Default values
MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET="ecqa"
ATTRIBUTION_METHOD="LIG"
NUM_EXPLANATIONS=5
USE_WANDB=true
RESUME_RUN=""
TEMPERATURE=0.7
SEED=42
INDICES_FILE="data/dataset_splits/kernel_shap_indices.json"

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_ID       Model ID to use (default: $MODEL_ID)"
    echo "  -d, --dataset DATASET      Dataset to use (default: $DATASET)"
    echo "  -a, --attribution METHOD   Attribution method to use (default: $ATTRIBUTION_METHOD)"
    echo "  -n, --num-exp NUMBER       Number of explanations per decision (default: $NUM_EXPLANATIONS)"
    echo "  --seed VALUE               Base seed for reproducible experiments (default: $SEED)"
    echo "  -w, --wandb                Enable wandb logging"
    echo "  --no-wandb                 Disable wandb logging"
    echo "  -r, --resume RUN_NAME      Resume a previous run"
    echo "  -g, --gpu GPU_ID           GPU ID to use (default: $CUDA_VISIBLE_DEVICES)"
    echo "  -t, --temperature VALUE    Temperature for model generation (default: $TEMPERATURE)"
    echo "  -i, --indices FILE         Kernel SHAP indices file (default: $INDICES_FILE)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset ecqa --attribution LIG --wandb"
    echo ""
    echo "This script will:"
    echo "  1. Load the kernel SHAP indices for the specified dataset"
    echo "  2. Filter the original dataset to only include those specific scenario IDs"
    echo "  3. Run the full data collection pipeline (decision + explanation + attribution)"
    echo "  4. Save results in data/kernel_shap_collection/ directory structure"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_ID="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -a|--attribution)
            ATTRIBUTION_METHOD="$2"
            shift 2
            ;;
        -n|--num-exp)
            NUM_EXPLANATIONS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -w|--wandb)
            USE_WANDB=true
            shift
            ;;
        --no-wandb)
            USE_WANDB=false
            shift
            ;;
        -r|--resume)
            RESUME_RUN="$2"
            shift 2
            ;;
        -g|--gpu)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -i|--indices)
            INDICES_FILE="$2"
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

# Activate environment if not already activated
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "ConstLLM" ]]; then
    echo "Activating ConstLLM conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ConstLLM || { echo "Failed to activate ConstLLM environment. Exiting."; exit 1; }
fi

# Check if we're in the correct directory (should have src/ directory)
if [[ ! -d "src" ]]; then
    echo "Error: Please run this script from the project root directory (where src/ directory is located)."
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if indices file exists
if [[ ! -f "$INDICES_FILE" ]]; then
    echo "Error: Kernel SHAP indices file '$INDICES_FILE' does not exist."
    echo "Please run the select_kernel_shap_indices.sh script first to generate the indices file."
    exit 1
fi

# Build command
CMD="python src/collect_data/run_kernel_shap_collection.py --model_id $MODEL_ID --dataset $DATASET --attribution_method $ATTRIBUTION_METHOD --num_dec_exp $NUM_EXPLANATIONS --temperature $TEMPERATURE --seed $SEED --indices_file $INDICES_FILE"

# Add optional parameters
if [[ "$USE_WANDB" == false ]]; then
    CMD="$CMD --no_wandb"
fi

if [[ -n "$RESUME_RUN" ]]; then
    CMD="$CMD --resume_run $RESUME_RUN"
fi

# Print command and configuration
echo "Kernel SHAP Data Collection Configuration:"
echo "  Model ID: $MODEL_ID"
echo "  Dataset: $DATASET"
echo "  Attribution Method: $ATTRIBUTION_METHOD"
echo "  Number of Explanations: $NUM_EXPLANATIONS"
echo "  Temperature: $TEMPERATURE"
echo "  Seed: $SEED"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  WandB: $USE_WANDB"
echo "  Indices File: $INDICES_FILE"
if [[ -n "$RESUME_RUN" ]]; then
    echo "  Resume Run: $RESUME_RUN"
fi
echo ""

# Print command
echo "Running command: $CMD"
echo "Starting kernel SHAP data collection at $(date)"
echo "------------------------------------"

# Run the command
$CMD

echo "------------------------------------"
echo "Kernel SHAP data collection completed at $(date)"
echo "Results saved in: data/kernel_shap_collection/$DATASET/"
