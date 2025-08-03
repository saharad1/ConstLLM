#!/bin/bash
#
# Data Collection Script for ConstLLM
# This script runs the data collection process for attribution analysis
#

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Exit on error
set -e

# GPU ID to use
# Set the GPU to use
export CUDA_VISIBLE_DEVICES=1

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

MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET="ecqa"
ATTRIBUTION_METHOD="LIG"
NUM_EXPLANATIONS=5
SUBSET=""
USE_WANDB=true
RESUME_RUN="ecqa_20250521_120325_LIG_llama3.1"
TEMPERATURE=0.7
SEED=42



# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_ID       Model ID to use (default: $MODEL_ID)"
    echo "  -d, --dataset DATASET      Dataset to use (default: $DATASET)"
    echo "  -a, --attribution METHOD   Attribution method to use (default: $ATTRIBUTION_METHOD)"
    echo "  -n, --num-exp NUMBER       Number of explanations per decision (default: $NUM_EXPLANATIONS)"
    echo "  -s, --subset SIZE          Size of dataset subset to use (default: all)"
    echo "  --seed VALUE               Base seed for reproducible experiments (default: $SEED)"
    echo "  -w, --wandb                Enable wandb logging"
    echo "  -r, --resume RUN_NAME      Resume a previous run"
    echo "  -g, --gpu GPU_ID           GPU ID to use (default: $GPU_ID)"
    echo "  -t, --temperature VALUE    Temperature for model generation (default: $TEMPERATURE)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset ecqa --wandb"
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
        -s|--subset)
            SUBSET="$2"
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
        -r|--resume)
            RESUME_RUN="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
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



# Build command
CMD="python -m src.collect_data.run_collect_data --model_id $MODEL_ID --dataset $DATASET --attribution_method $ATTRIBUTION_METHOD --num_dec_exp $NUM_EXPLANATIONS --temperature $TEMPERATURE --seed $SEED"

# Add optional parameters
if [[ -n "$SUBSET" && "$SUBSET" != "None" ]]; then
    CMD="$CMD --subset $SUBSET"
fi

if [[ "$USE_WANDB" == false ]]; then
    CMD="$CMD --no_wandb"
fi

if [[ -n "$RESUME_RUN" ]]; then
    CMD="$CMD --resume_run $RESUME_RUN"
fi

# Print command
echo "Running command: $CMD"
echo "Starting data collection at $(date)"
echo "------------------------------------"

# Run the command
$CMD

echo "------------------------------------"
echo "Data collection completed at $(date)"