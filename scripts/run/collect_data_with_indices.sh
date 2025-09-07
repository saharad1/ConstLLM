#!/bin/bash
#
# Data Collection Script with Dataset Indices for ConstLLM
# This script runs the data collection process using pre-selected scenario IDs
# using pre-selected scenario IDs from the test sets
# (Same functionality as collect_data.sh but with indices file filtering)
#

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Exit on error
set -e

# This script focuses on execution - model list selection is handled by launch scripts


# Attribution methods:
# LIG
# LIME
# Feature Ablation
# KSHAP

# Datasets:
# ecqa
# choice75
# codah
# arc_easy
# arc_challenge

# Variables - no defaults, all values provided by launcher
MODEL_ID=""
DATASET=""
ATTRIBUTION_METHOD=""
NUM_EXPLANATIONS=""
SUBSET=""
USE_WANDB=""
RESUME_RUN=""
TEMPERATURE=""
SEED=""
INDICES_FILE=""
MODEL_LIST=""
USE_MODEL_LIST=false

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_ID       Model ID to use (for single model execution)"
    echo "  -c, --custom-list MODELS   Custom space-separated list of model IDs (for multiple models)"
    echo "  -d, --dataset DATASET      Dataset to use"
    echo "  -a, --attribution METHOD   Attribution method to use"
    echo "  -n, --num-exp NUMBER       Number of explanations per decision"
    echo "  -s, --subset SIZE          Size of dataset subset to use"
    echo "  --seed VALUE               Base seed for reproducible experiments"
    echo "  -w, --wandb                Enable wandb logging"
    echo "  -r, --resume RUN_NAME      Resume a previous run"
    echo "  -g, --gpu GPU_ID           GPU ID to use"
    echo "  -t, --temperature VALUE    Temperature for model generation"
    echo "  -i, --indices FILE         Dataset indices file"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Single model:"
    echo "  $0 --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset ecqa --wandb"
    echo ""
    echo "  # Multiple models:"
    echo "  $0 --custom-list \"model1 model2 model3\" --dataset ecqa --wandb"
    echo ""
    echo "Note: For predefined model lists, use the launch script:"
    echo "  scripts/launch/launch_collect_data_with_indices.sh --model-list all_models"
    echo ""
    echo "This script will:"
    echo "  1. Load the dataset indices for the specified dataset"
    echo "  2. Filter the original dataset to only include those specific scenario IDs"
    echo "  3. Run the full data collection pipeline (decision + explanation + attribution) for each model"
    echo "  4. Save results in data/collect_data_with_indices/ directory structure"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_ID="$2"
            USE_MODEL_LIST=false
            shift 2
            ;;
        -c|--custom-list)
            MODEL_LIST="$2"
            USE_MODEL_LIST=true
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

# Check if either single model or model list is provided
if [[ -z "$MODEL_ID" && -z "$MODEL_LIST" ]]; then
    echo "Error: You must specify either --model (for single model) or --custom-list (for multiple models)."
    echo "Use the launch script for predefined model lists:"
    echo "  scripts/launch/launch_collect_data_with_indices.sh all_models"
    exit 1
fi

# Check if all required parameters are provided
if [[ -z "$DATASET" || -z "$ATTRIBUTION_METHOD" || -z "$NUM_EXPLANATIONS" || -z "$TEMPERATURE" || -z "$SEED" || -z "$INDICES_FILE" ]]; then
    echo "Error: Missing required parameters. All parameters must be provided."
    echo "Use the launch script which provides all required parameters:"
    echo "  scripts/launch/launch_collect_data_with_indices.sh all_models"
    exit 1
fi

# Check if indices file exists
if [[ ! -f "$INDICES_FILE" ]]; then
    echo "Error: Dataset indices file '$INDICES_FILE' does not exist."
    echo "Please run the select_datasets_indices.sh script first to generate the indices file."
    exit 1
fi

# Function to build command for a single model
build_command() {
    local model_id="$1"
    local cmd="python -m src.collect_data.run_collect_data_with_indices --model_id $model_id --dataset $DATASET --attribution_method $ATTRIBUTION_METHOD --num_dec_exp $NUM_EXPLANATIONS --temperature $TEMPERATURE --seed $SEED --indices_file $INDICES_FILE"
    
    # Add optional parameters
    if [[ -n "$SUBSET" && "$SUBSET" != "None" ]]; then
        cmd="$cmd --subset $SUBSET"
    fi
    
    if [[ "$USE_WANDB" == false ]]; then
        cmd="$cmd --no_wandb"
    fi
    
    if [[ -n "$RESUME_RUN" ]]; then
        cmd="$cmd --resume_run $RESUME_RUN"
    fi
    
    echo "$cmd"
}

# Function to run command for a single model
run_single_model() {
    local model_id="$1"
    local model_num="$2"
    local total_models="$3"
    
    echo "=========================================="
    echo "Running model $model_num/$total_models: $model_id"
    echo "Started at: $(date)"
    echo "=========================================="
    
    local cmd=$(build_command "$model_id")
    echo "Command: $cmd"
    echo ""
    
    # Run the command
    if $cmd; then
        echo ""
        echo "✅ Model $model_num/$total_models completed successfully: $model_id"
        echo "Completed at: $(date)"
    else
        echo ""
        echo "❌ Model $model_num/$total_models failed: $model_id"
        echo "Failed at: $(date)"
        echo "Continuing with next model..."
    fi
    echo ""
}

# Main execution logic
echo "Starting collect data with indices at $(date)"
echo "=========================================="

if [[ "$USE_MODEL_LIST" == true ]]; then
    # Convert MODEL_LIST to array
    read -ra MODELS <<< "$MODEL_LIST"
    total_models=${#MODELS[@]}
    
    echo "Running data collection for $total_models models:"
    for i in "${!MODELS[@]}"; do
        echo "  $((i+1)). ${MODELS[i]}"
    done
    echo ""
    
    # Loop through each model
    for i in "${!MODELS[@]}"; do
        model_num=$((i+1))
        run_single_model "${MODELS[i]}" "$model_num" "$total_models"
    done
    
    echo "=========================================="
    echo "All models completed at $(date)"
    echo "Results saved in: data/collect_data_with_indices/$DATASET/"
else
    # Single model execution (original behavior)
    echo "Running data collection for single model: $MODEL_ID"
    echo ""
    
    cmd=$(build_command "$MODEL_ID")
    echo "Command: $cmd"
    echo ""
    
    if $cmd; then
        echo ""
        echo "✅ Single model completed successfully: $MODEL_ID"
    else
        echo ""
        echo "❌ Single model failed: $MODEL_ID"
        exit 1
    fi
    
    echo "=========================================="
    echo "Collect data with indices completed at $(date)"
    echo "Results saved in: data/collect_data_with_indices/$DATASET/"
fi
