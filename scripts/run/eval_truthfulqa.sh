#!/bin/bash
#
# TruthfulQA Evaluation Script for ConstLLM
# This script runs TruthfulQA evaluation on your chosen models
#

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Exit on error
set -e

# Set default GPU
export CUDA_VISIBLE_DEVICES=1

# ==============================================
# CORE PARAMETERS (Essential for all tasks)
# ==============================================
# Default models to evaluate (can be overridden with --models)
DEFAULT_MODELS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "unsloth/Meta-Llama-3.1-8B-Instruct"
    "unsloth/Llama-3.2-3B-Instruct"
)

# Trained LIME models to evaluate
TRAINED_LIME_MODELS=(
    "models/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_250510_194800_lr4.65e-06_beta5.64/checkpoint-448"
    "models/arc_easy/Llama-3.2-3B-Instruct/arc_easy_250516_015641_lr6.32e-06_beta8.84/best_model"
    "models/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_250508_224902_lr4.21e-06_beta5.13/best_model"
    "models/ecqa/Llama-3.2-3B-Instruct/ecqa_250513_133025_lr9.55e-06_beta8.44/best_model"
)

# Trained LIG models to evaluate
TRAINED_LIG_MODELS=(
    "models/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_250806_181728_lr6.86e-06_beta8.41/best_model"
    "models/arc_easy/Llama-3.2-3B-Instruct/arc_easy_250805_195416_lr3.84e-06_beta9.20/best_model"
    "models/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_250818_054203_lr8.96e-06_beta7.04/best_model"
    "models/ecqa/Llama-3.2-3B-Instruct/ecqa_250808_050834_lr9.56e-06_beta9.93/best_model"
)

# single_model="models/ecqa/Llama-3.2-3B-Instruct/ecqa_250513_133025_lr9.55e-06_beta8.44/best_model"

# Combine all models by default
ALL_MODELS=("${DEFAULT_MODELS[@]}" "${TRAINED_LIG_MODELS[@]}" "${TRAINED_LIME_MODELS[@]}")

# MODELS=("${single_model}")
MODELS=("${ALL_MODELS[@]}")  # Array of models to evaluate
TASK="mc"  # Options: mc, generation, both (start with 'mc' for simplicity)
TEMPERATURE=0.0  # Temperature for YOUR model (0.0 = deterministic)
OUTPUT_DIR="results/truthfulqa"  # Where to save results
GPU_ID=1  # Which GPU to use
DEVICE_MAP="auto"  # Multi-GPU device mapping

# ==============================================
# OPTIONAL PARAMETERS (Only needed for specific tasks)
# ==============================================

# For Multiple-Choice Task Limits (optional - leave empty for all questions)
NUM_MC_QUESTIONS=""  # e.g., "100" to test only 100 questions

# For Generation Task (only needed if TASK="generation" or "both")
NUM_GEN_QUESTIONS=""  # Number of generation questions (empty = all)
MAX_NEW_TOKENS=50  # Max tokens YOUR model generates per answer
TRUTHFULNESS_JUDGE="heuristic"  # How to judge answers: heuristic, gpt-3.5-turbo, gpt-4

# For GPT-based Judging (only needed if TRUTHFULNESS_JUDGE starts with "gpt")
OPENAI_API_KEY=""  # OpenAI API key for external judges

# Advanced Options
COMPREHENSIVE_EVAL=true  # Use comprehensive evaluation script

# Function to create smart model name for folder organization
function get_smart_model_name {
    local model_path="$1"
    
    # Check if it's a trained model (contains 'models/' and training info)
    if [[ "$model_path" == models/* ]]; then
        # Extract dataset, base model, and training timestamp
        # Example: models/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_250806_181728_lr6.86e-06_beta8.41/best_model
        # Should become: arc_easy_Meta-Llama-3.1-8B-Instruct_250806_181728
        
        # Extract dataset name (first part after models/)
        local dataset=$(echo "$model_path" | sed 's|models/\([^/]*\)/.*|\1|')
        
        # Extract base model name
        local base_model=$(echo "$model_path" | sed 's|.*/\([^/]*\)/[^/]*/.*|\1|')
        
        # Extract timestamp (format: YYMMDD_HHMMSS)
        local timestamp=$(echo "$model_path" | grep -o '[0-9]\{6\}_[0-9]\{6\}' | head -1)
        
        # Extract learning rate and beta for uniqueness
        local lr=$(echo "$model_path" | grep -o 'lr[0-9.]*e-[0-9]*' | head -1)
        local beta=$(echo "$model_path" | grep -o 'beta[0-9.]*' | head -1)
        
        if [[ -n "$dataset" && -n "$base_model" && -n "$timestamp" ]]; then
            echo "${dataset}_${base_model}_${timestamp}_${lr}_${beta}"
        else
            # Fallback: use the last part of the path
            basename "$model_path" | tr '/' '_'
        fi
    else
        # For default models, use a clean name
        # Example: meta-llama/Meta-Llama-3.1-8B-Instruct -> Meta-Llama-3.1-8B-Instruct
        # Example: unsloth/Meta-Llama-3.1-8B-Instruct -> unsloth_Meta-Llama-3.1-8B-Instruct
        if [[ "$model_path" == *"/"* ]]; then
            # Has organization prefix
            local org=$(echo "$model_path" | cut -d'/' -f1)
            local model=$(echo "$model_path" | cut -d'/' -f2-)
            echo "${org}_${model}"
        else
            # No organization prefix
            echo "$model_path"
        fi
    fi
}

# Function to get model category for folder organization
function get_model_category {
    local model_path="$1"
    
    if [[ "$model_path" == models/* ]]; then
        echo "trained"
    else
        echo "default"
    fi
}

# Display help message
function show_help {
    echo "TruthfulQA Evaluation Script"
    echo "============================"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "=== CORE OPTIONS (Essential) ==="
    echo "  -m, --model MODEL_ID         Single model to evaluate (legacy option)"
    echo "  --models MODEL1 MODEL2 ...   Multiple models to evaluate"
    echo "  --models-file FILE           File containing model IDs (one per line)"
    echo "  --default-models             Evaluate only default models"
    echo "  --trained-lig-models             Evaluate only trained models"
    echo "  --trained-lime-models             Evaluate only trained LIME models"
    echo "  --all-models                 Evaluate all models (default)"
    echo "  -t, --task TASK              Task: mc, generation, both (default: $TASK)"
    echo "  --temperature VALUE          Temperature for YOUR model (default: $TEMPERATURE)"
    echo "  -g, --gpu GPU_ID             GPU ID to use (default: $GPU_ID)"
    echo "  -o, --output-dir DIR         Output directory (default: $OUTPUT_DIR)"
    echo ""
    echo "=== OPTIONAL LIMITS ==="
    echo "  --num-mc NUMBER              Limit MC questions (default: all)"
    echo "  --num-gen NUMBER             Limit generation questions (default: all)"
    echo ""
    echo "=== GENERATION TASK OPTIONS (only if task=generation/both) ==="
    echo "  --max-tokens NUMBER          Max tokens YOUR model generates (default: $MAX_NEW_TOKENS)"
    echo "  --judge JUDGE                Answer judge: heuristic, gpt-3.5-turbo, gpt-4 (default: $TRUTHFULNESS_JUDGE)"
    echo "  --openai-key KEY             OpenAI key (only needed for GPT judges)"
    echo ""
    echo "=== ADVANCED OPTIONS ==="
    echo "  --device-map MAP             Multi-GPU device mapping (default: $DEVICE_MAP)"
    echo "  --simple                     Use simple evaluation script instead of comprehensive"
    echo ""
    echo "Other Options:"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Available Model Categories:"
    echo "  Default Models:"
    for model in "${DEFAULT_MODELS[@]}"; do
        echo "    - $model"
    done
    echo ""
    echo "  Trained Models:"
    for model in "${TRAINED_LIG_MODELS[@]}"; do
        echo "    - $model"
    done
    echo ""
    echo "  Trained LIME Models:"
    for model in "${TRAINED_LIME_MODELS[@]}"; do
        echo "    - $model"
    done
    echo ""
    echo "Examples:"
    echo "  # Evaluate all models (default)"
    echo "  $0 --task mc --num-mc 50"
    echo ""
    echo "  # Evaluate only default models"
    echo "  $0 --default-models --task mc"
    echo ""
    echo "  # Evaluate only trained models"
    echo "  $0 --trained-lig-models --task both"
    echo ""
    echo "  # Evaluate only trained LIME models"
    echo "  $0 --trained-lime-models --task both"
    echo ""
    echo "  # Evaluate all models"
    echo "  $0 --all-models --task both"
    echo ""
    echo "  # Evaluate specific models"
    echo "  $0 --models meta-llama/Meta-Llama-3.1-8B-Instruct gpt2 --task mc"
    echo ""
    echo "  # Single model (legacy)"
    echo "  $0 --model meta-llama/Meta-Llama-3.1-8B-Instruct --task mc"
    echo ""
    exit 0
}

# Function to read models from file
function read_models_from_file {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "Error: Models file '$file' not found"
        exit 1
    fi
    
    # Read models from file, skipping empty lines and comments
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
            MODELS+=("$line")
        fi
    done < "$file"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            # Legacy single model option
            MODELS=("$2")
            shift 2
            ;;
        --models)
            # Multiple models option
            shift  # Remove --models
            MODELS=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --models-file)
            MODELS=()  # Clear default models
            read_models_from_file "$2"
            shift 2
            ;;
        --default-models)
            MODELS=("${DEFAULT_MODELS[@]}")
            shift
            ;;
        --trained-lig-models)
            MODELS=("${TRAINED_LIG_MODELS[@]}")
            shift
            ;;
        --trained-lime-models)
            MODELS=("${TRAINED_LIME_MODELS[@]}")
            shift
            ;;
        --all-models)
            MODELS=("${ALL_MODELS[@]}")
            shift
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-mc)
            NUM_MC_QUESTIONS="$2"
            shift 2
            ;;
        --num-gen)
            NUM_GEN_QUESTIONS="$2"
            shift 2
            ;;
        --judge)
            TRUTHFULNESS_JUDGE="$2"
            shift 2
            ;;
        --openai-key)
            OPENAI_API_KEY="$2"
            shift 2
            ;;
        --device-map)
            DEVICE_MAP="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --simple)
            COMPREHENSIVE_EVAL=false
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Validate task
if [[ ! "$TASK" =~ ^(mc|generation|both)$ ]]; then
    echo "Error: Task must be 'mc', 'generation', or 'both'"
    exit 1
fi

# Validate judge for generation tasks
if [[ "$TASK" == "generation" || "$TASK" == "both" ]]; then
    if [[ ! "$TRUTHFULNESS_JUDGE" =~ ^(heuristic|gpt-3.5-turbo|gpt-4)$ ]]; then
        echo "Error: Truthfulness judge must be 'heuristic', 'gpt-3.5-turbo', or 'gpt-4'"
        exit 1
    fi
    
    if [[ "$TRUTHFULNESS_JUDGE" =~ ^gpt && -z "$OPENAI_API_KEY" ]]; then
        echo "Warning: GPT judge selected but no OpenAI API key provided. Using heuristic judge."
        TRUTHFULNESS_JUDGE="heuristic"
    fi
fi

# Activate environment if not already activated
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "ConstLLM" ]]; then
    echo "Activating ConstLLM conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ConstLLM || { echo "Failed to activate ConstLLM environment. Exiting."; exit 1; }
fi

# Create output directory with organized structure
mkdir -p "$OUTPUT_DIR"

# Display configuration
echo "==============================================="
echo "TruthfulQA Evaluation Configuration"
echo "==============================================="
echo "Models to evaluate:"
for model in "${MODELS[@]}"; do
    category=$(get_model_category "$model")
    smart_name=$(get_smart_model_name "$model")
    echo "  - [$category] $model -> $smart_name"
done
echo "Task: $TASK"
echo "GPU: $GPU_ID"
echo "Device Map: $DEVICE_MAP"
echo "Temperature: $TEMPERATURE"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Output Directory: $OUTPUT_DIR"
if [[ -n "$NUM_MC_QUESTIONS" ]]; then
    echo "MC Questions: $NUM_MC_QUESTIONS"
fi
if [[ -n "$NUM_GEN_QUESTIONS" ]]; then
    echo "Generation Questions: $NUM_GEN_QUESTIONS"
fi
if [[ "$TASK" == "generation" || "$TASK" == "both" ]]; then
    echo "Truthfulness Judge: $TRUTHFULNESS_JUDGE"
fi
echo "Comprehensive Evaluation: $COMPREHENSIVE_EVAL"
echo "==============================================="
echo ""

# Function to evaluate a single model
function evaluate_model {
    local model_id="$1"
    local smart_name=$(get_smart_model_name "$model_id")
    local category=$(get_model_category "$model_id")
    
    # Create category-specific output directory
    local model_output_dir="$OUTPUT_DIR/${category}/${smart_name}"
    mkdir -p "$model_output_dir"
    
    echo "==============================================="
    echo "Evaluating model: $model_id"
    echo "Category: $category"
    echo "Smart name: $smart_name"
    echo "Output dir: $model_output_dir"
    echo "==============================================="
    echo "Started at: $(date)"
    echo ""
    
    # Build command for this model
    if [[ "$COMPREHENSIVE_EVAL" == true ]]; then
        # Use comprehensive evaluation script
        CMD="PYTHONPATH=/raid/saharad/ConstLLM python src/truthfulqa_eval/truthfulqa_comprehensive_eval.py --model_name \"$model_id\" --task $TASK --temperature $TEMPERATURE --max_new_tokens $MAX_NEW_TOKENS --output_dir \"$model_output_dir\" --device_map $DEVICE_MAP"
        
        # Add optional parameters
        if [[ -n "$NUM_MC_QUESTIONS" ]]; then
            CMD="$CMD --num_mc_questions $NUM_MC_QUESTIONS"
        fi
        
        if [[ -n "$NUM_GEN_QUESTIONS" ]]; then
            CMD="$CMD --num_gen_questions $NUM_GEN_QUESTIONS"
        fi

    elif [[ "$TASK" == "generation" ]]; then
        # Use specialized generation script
        CMD="PYTHONPATH=/raid/saharad/ConstLLM python src/truthfulqa_generation_eval.py --model_name \"$model_id\" --temperature $TEMPERATURE --max_new_tokens $MAX_NEW_TOKENS --output_dir \"$model_output_dir\" --truthfulness_judge $TRUTHFULNESS_JUDGE"
        
        if [[ -n "$NUM_GEN_QUESTIONS" ]]; then
            CMD="$CMD --num_questions $NUM_GEN_QUESTIONS"
        fi
        
        if [[ -n "$OPENAI_API_KEY" ]]; then
            CMD="$CMD --openai_api_key \"$OPENAI_API_KEY\""
        fi

    else
        # Use comprehensive for MC-only
        CMD="PYTHONPATH=/raid/saharad/ConstLLM python src/truthfulqa_comprehensive_eval.py --model_name \"$model_id\" --task mc --temperature $TEMPERATURE --output_dir \"$model_output_dir\" --device_map $DEVICE_MAP"
        
        if [[ -n "$NUM_MC_QUESTIONS" ]]; then
            CMD="$CMD --num_mc_questions $NUM_MC_QUESTIONS"
        fi
    fi

    # Run the evaluation
    echo "Command: $CMD"
    echo ""
    
    if eval $CMD; then
        echo ""
        echo "✅ Model $model_id evaluation completed successfully!"
        echo "Completed at: $(date)"
        echo ""
    else
        echo ""
        echo "❌ Model $model_id evaluation failed!"
        echo "Failed at: $(date)"
        echo ""
    fi
    
    echo "==============================================="
    echo ""
}

# Evaluate all models
echo "Starting evaluation of ${#MODELS[@]} models..."
echo ""

for model in "${MODELS[@]}"; do
    evaluate_model "$model"
done

# Generate summary report
echo "==============================================="
echo "Evaluation Summary"
echo "==============================================="
echo "All models evaluated at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Show organized result files
echo "Generated result files (organized by category):"
echo "Default models:"
find "$OUTPUT_DIR/default" -name "*.json" -type f 2>/dev/null | sort || echo "  No default model results found"
echo ""
echo "Trained models:"
find "$OUTPUT_DIR/trained" -name "*.json" -type f 2>/dev/null | sort || echo "  No trained model results found"
echo ""

# Try to show quick summary if results exist
echo "Quick Summary:"
echo "--------------"
for model in "${MODELS[@]}"; do
    smart_name=$(get_smart_model_name "$model")
    category=$(get_model_category "$model")
    model_output_dir="$OUTPUT_DIR/${category}/${smart_name}"
    result_file=$(find "$model_output_dir" -name "*.json" | head -1)
    
    if [[ -n "$result_file" && -f "$result_file" ]]; then
        echo "Model: $model"
        echo "Category: $category"
        echo "Smart name: $smart_name"
        # Extract key metrics using jq if available, otherwise use grep/sed
        if command -v jq >/dev/null 2>&1; then
            MC_ACC=$(jq -r '.metrics.mc1_accuracy // .metrics.multiple_choice.mc1_accuracy // "N/A"' "$result_file" 2>/dev/null)
            TRUTH_PCT=$(jq -r '.metrics.truthfulness_percentage // .metrics.generation.truthfulness_percentage // "N/A"' "$result_file" 2>/dev/null)
            INFO_PCT=$(jq -r '.metrics.informativeness_percentage // .metrics.generation.informativeness_percentage // "N/A"' "$result_file" 2>/dev/null)
            
            if [[ "$MC_ACC" != "N/A" && "$MC_ACC" != "null" ]]; then
                echo "  Multiple Choice Accuracy: ${MC_ACC}%"
            fi
            if [[ "$TRUTH_PCT" != "N/A" && "$TRUTH_PCT" != "null" ]]; then
                echo "  Truthfulness: ${TRUTH_PCT}%"
                echo "  Informativeness: ${INFO_PCT}%"
            fi
        else
            echo "  Results available in: $result_file"
        fi
        echo ""
    else
        echo "Model: $model - No results found"
        echo ""
    fi
done

echo "To analyze results further:"
echo "python src/truthfulqa_utils.py analyze --results_dir \"$OUTPUT_DIR\""
echo ""