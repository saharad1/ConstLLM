#!/bin/bash
#
# Configuration and Launch Script for Data Collection with Indices
# This script handles all configuration and launches the main collection script
#

# GPU selection is controlled via the -g/--gpu flag
# (do not set CUDA_VISIBLE_DEVICES globally here)

# Predefined model lists
declare -A MODEL_LISTS

# Model ids:
# unsloth/mistral-7b-instruct-v0.3
# unsloth/Meta-Llama-3.1-8B-Instruct
# unsloth/Qwen2.5-7B-Instruct
# unsloth/Llama-3.2-3B-Instruct

# For LIG:
# meta-llama/Meta-Llama-3.1-8B-Instruct
# mistralai/Mistral-7B-Instruct-v0.3
# meta-llama/Llama-3.2-3B-Instruct

MODEL_LISTS["lig_models"]="meta-llama/Meta-Llama-3.1-8B-Instruct \
meta-llama/Llama-3.2-3B-Instruct \
models/arc_easy/Llama-3.2-3B-Instruct/arc_easy_250805_195416_lr3.84e-06_beta9.20/best_model \
models/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_250806_181728_lr6.86e-06_beta8.41/best_model \
models/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_250818_054203_lr8.96e-06_beta7.04/best_model \
models/ecqa/Llama-3.2-3B-Instruct/ecqa_250808_050834_lr9.56e-06_beta9.93/best_model"

MODEL_LISTS["lime_models"]="unsloth/Meta-Llama-3.1-8B-Instruct \
unsloth/Llama-3.2-3B-Instruct \
models/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_250510_194800_lr4.65e-06_beta5.64/checkpoint-448 \
models/arc_easy/Llama-3.2-3B-Instruct/arc_easy_250516_015641_lr6.32e-06_beta8.84/best_model \
models/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_250508_224902_lr4.21e-06_beta5.13/best_model \
models/ecqa/Llama-3.2-3B-Instruct/ecqa_250513_133025_lr9.55e-06_beta8.44/best_model"


# Configuration - Modify these values as needed

# Model configuration (choose one):
# Option 1: Use a predefined model list
MODEL_LIST="lime_models"  # Set to: lig_models, lime_models

# Option 2: Use a single model (uncomment and set)
# SINGLE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Other configuration
DATASET="ecqa"
ATTRIBUTION_METHOD="KSHAP"
NUM_EXPLANATIONS=5
SUBSET=""
USE_WANDB=true
RESUME_RUN=""
TEMPERATURE=0.7
SEED=42
INDICES_FILE="data/dataset_splits/datasets_test_indices.json"
GPU_ID=2



# Display help message
function show_help {
    echo "Usage: $0 [model_or_list] [options]"
    echo ""
    echo "Model specification:"
    echo "  Single model:     $0 --model MODEL_ID"
    echo "  Model list:       $0 MODEL_LIST_NAME"
    echo "  Custom list:      $0 --custom-list \"model1 model2 model3\""
    echo ""
    echo "Available model lists:"
    echo "  lig_models        - All LIG-compatible models"
    echo "  lime_models       - All LIME-compatible models"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_ID       Single model ID to use"
    echo "  -c, --custom-list MODELS   Custom space-separated list of model IDs"
    echo "  -d, --dataset DATASET      Dataset to use (default: $DATASET)"
    echo "  -a, --attribution METHOD   Attribution method to use (default: $ATTRIBUTION_METHOD)"
    echo "  -n, --num-exp NUMBER       Number of explanations per decision (default: $NUM_EXPLANATIONS)"
    echo "  -s, --subset SIZE          Size of dataset subset to use (default: all)"
    echo "  --seed VALUE               Base seed for reproducible experiments (default: $SEED)"
    echo "  -w, --wandb                Enable wandb logging (default: $USE_WANDB)"
    echo "  -r, --resume RUN_NAME      Resume a previous run"
    echo "  -g, --gpu GPU_ID           GPU ID to use (default: $GPU_ID)"
    echo "  -t, --temperature VALUE    Temperature for model generation (default: $TEMPERATURE)"
    echo "  -i, --indices FILE         Dataset indices file (default: $INDICES_FILE)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Single model:"
    echo "  $0 --model meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "  $0 --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset ecqa"
    echo ""
    echo "  # Model list:"
    echo "  $0 lig_models"
    echo "  $0 lig_models --dataset ecqa --attribution LIG --wandb"
    echo ""
    echo "  # Custom list:"
    echo "  $0 --custom-list \"model1 model2 model3\" --dataset ecqa"
    exit 0
}

# Parse command line arguments
MODEL_ID=""
MODEL_LIST_NAME=""
CUSTOM_MODEL_LIST=""
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_ID="$2"
            MODE="single"
            shift 2
            ;;
        -c|--custom-list)
            CUSTOM_MODEL_LIST="$2"
            MODE="custom"
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
        -i|--indices)
            INDICES_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            ;;
        *)
            if [[ -z "$MODEL_LIST_NAME" && -z "$MODE" ]]; then
                MODEL_LIST_NAME="$1"
                MODE="list"
            else
                echo "Error: Multiple model specifications. Use only one of: --model, --custom-list, or model list name."
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if model specification is provided, use configuration if not
if [[ -z "$MODE" ]]; then
    # Use configured model specification
    if [[ -n "$MODEL_LIST" ]]; then
        MODEL_LIST_NAME="$MODEL_LIST"
        MODE="list"
        echo "Using configured model list: $MODEL_LIST"
    elif [[ -n "$SINGLE_MODEL" ]]; then
        MODEL_ID="$SINGLE_MODEL"
        MODE="single"
        echo "Using configured single model: $SINGLE_MODEL"
    elif [[ -n "$CUSTOM_MODEL_LIST" ]]; then
        CUSTOM_MODEL_LIST="$CUSTOM_MODEL_LIST"
        MODE="custom"
        echo "Using configured custom list: $CUSTOM_MODEL_LIST"
    else
        echo "Error: You must specify a model. Use one of:"
        echo "  --model MODEL_ID"
        echo "  --custom-list \"model1 model2 model3\""
        echo "  MODEL_LIST_NAME (e.g., all_models)"
        echo ""
        echo "Or configure a model in the configuration section:"
        echo "  MODEL_LIST=\"all_models\""
        echo "  SINGLE_MODEL=\"meta-llama/Meta-Llama-3.1-8B-Instruct\""
        echo "  CUSTOM_MODEL_LIST=\"model1 model2 model3\""
        echo ""
        show_help
    fi
fi

# Resolve the model list based on mode
case "$MODE" in
    "single")
        MODEL_LIST="$MODEL_ID"
        DISPLAY_NAME="Single model: $MODEL_ID"
        ;;
    "custom")
        MODEL_LIST="$CUSTOM_MODEL_LIST"
        DISPLAY_NAME="Custom list ($(echo $CUSTOM_MODEL_LIST | wc -w) models)"
        ;;
    "list")
        # Validate model list
        if [[ -z "${MODEL_LISTS[$MODEL_LIST_NAME]}" ]]; then
            echo "Error: Unknown model list '$MODEL_LIST_NAME'"
            echo "Available lists: ${!MODEL_LISTS[@]}"
            exit 1
        fi
        MODEL_LIST="${MODEL_LISTS[$MODEL_LIST_NAME]}"
        DISPLAY_NAME="Model list: $MODEL_LIST_NAME ($(echo $MODEL_LIST | wc -w) models)"
        ;;
esac

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN_SCRIPT="$PROJECT_ROOT/scripts/run/collect_data_with_indices.sh"

# Check if main script exists
if [[ ! -f "$MAIN_SCRIPT" ]]; then
    echo "Error: Main collection script not found at $MAIN_SCRIPT"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Build command with all configured parameters
if [[ "$MODE" == "single" ]]; then
    CMD="bash $MAIN_SCRIPT --model \"$MODEL_LIST\" --dataset $DATASET --attribution $ATTRIBUTION_METHOD --num-exp $NUM_EXPLANATIONS --temperature $TEMPERATURE --seed $SEED --indices $INDICES_FILE --gpu $GPU_ID"
else
    CMD="bash $MAIN_SCRIPT --custom-list \"$MODEL_LIST\" --dataset $DATASET --attribution $ATTRIBUTION_METHOD --num-exp $NUM_EXPLANATIONS --temperature $TEMPERATURE --seed $SEED --indices $INDICES_FILE --gpu $GPU_ID"
fi

# Add optional parameters
if [[ -n "$SUBSET" && "$SUBSET" != "None" ]]; then
    CMD="$CMD --subset $SUBSET"
fi

if [[ "$USE_WANDB" == true ]]; then
    CMD="$CMD --wandb"
fi

if [[ -n "$RESUME_RUN" ]]; then
    CMD="$CMD --resume $RESUME_RUN"
fi

# Display configuration
echo "🚀 Launching Data Collection with Indices"
echo "=========================================="
echo "📋 $DISPLAY_NAME"
echo "📊 Dataset: $DATASET"
echo "🔍 Attribution: $ATTRIBUTION_METHOD"
echo "📝 Explanations: $NUM_EXPLANATIONS"
echo "🌡️  Temperature: $TEMPERATURE"
echo "🎲 Seed: $SEED"
echo "🖥️  GPU: $GPU_ID"
echo "📈 Wandb: $USE_WANDB"
if [[ -n "$SUBSET" && "$SUBSET" != "None" ]]; then
    echo "📏 Subset: $SUBSET"
fi
if [[ -n "$RESUME_RUN" ]]; then
    echo "🔄 Resume: $RESUME_RUN"
fi
echo "=========================================="
echo ""

# Execute
echo "⚙️  Command: $CMD"
echo ""
eval $CMD
