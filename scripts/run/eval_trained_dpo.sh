#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Set environment variables if needed
# Only set CUDA_VISIBLE_DEVICES if it's not already set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# Out of domain datasets:
# codah,llama3.1: data/collection_data/codah/unsloth_Meta-Llama-3.1-8B-Instruct/codah_20250415_125210_LIME_llama3.1/test_278.jsonl
# codah,llama3.2: data/collection_data/codah/unsloth_Llama-3.2-3B-Instruct/codah_20250506_085629_LIME_llama3.2/test_272.jsonl
# arc_challenge,llama3.1: data/collection_data/arc_challenge/unsloth_Meta-Llama-3.1-8B-Instruct/arc_challenge_20250415_155822_LIME_llama3.1/test_260.jsonl
# arc_challenge,llama3.2: data/collection_data/arc_challenge/unsloth_Llama-3.2-3B-Instruct/arc_challenge_20250421_094925_LIME_llama3.2/test_255.jsonl

# Out of domain models:
# ecqa,llama3.1: models/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_250508_224902_lr4.21e-06_beta5.13/best_model
# ecqa,llama3.2: models/ecqa/Llama-3.2-3B-Instruct/ecqa_250513_133025_lr9.55e-06_beta8.44/best_model
# arc_easy,llama3.1: models/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_250510_194800_lr4.65e-06_beta5.64/checkpoint-448
# arc_easy,llama3.2: models/arc_easy/Llama-3.2-3B-Instruct/arc_easy_250516_015641_lr6.32e-06_beta8.84/best_model

# Default paths - adjust these as needed
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_PATH="data/collection_data/arc_easy/meta-llama_Meta-Llama-3.1-8B-Instruct/arc_easy_20250527_100818_LIG_llama3.1/test_521.jsonl"
OUTPUT_DIR=""
RESUME_RUN=""

IGNORE_PRE_GENERATED=true

# Better detection of model ID vs local path
if [[ "$MODEL_PATH" =~ ^[./] || "$MODEL_PATH" =~ ^/ || 
      $(echo "$MODEL_PATH" | tr -cd '/' | wc -c) -gt 1 || 
      "$MODEL_PATH" =~ ^trained_models ]]; then
  # Likely a local path
  IS_MODEL_ID=""
else
  # Likely a model ID (org/model format with one slash)
  IS_MODEL_ID="--is_model_id"
fi

# Attribution Methods:
# LIG
# LIME


# Attribution method and other parameters
ATTRIBUTION_METHOD="LIG"
NUM_DEC_EXP=5
TEMPERATURE=0.7
SUBSET=""
SEED=42


# Default is to disable wandb (set to false)
USE_WANDB=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --model_path|-m     Path to the trained model directory or HuggingFace model ID"
      echo "  --dataset_path|-d   Path to the dataset file (JSONL format)"
      echo "  --attribution_method|-a  Attribution method to use (default: $ATTRIBUTION_METHOD)"
      echo "  --num_dec_exp|-n    Number of explanations per decision (default: $NUM_DEC_EXP)"
      echo "  --temperature|-t    Temperature for model generation (default: $TEMPERATURE)"
      echo "  --output_dir|-o     Custom output directory for results"
      echo "  --wandb|-w          Enable wandb logging"
      echo "  --subset|-s         Size of dataset subset to use"
      echo "  --seed              Base seed for reproducible experiments (default: $SEED)"
      echo "  --ignore_pre_generated  Ignore any pre-generated attributions in the dataset"
      echo "  --is_model_id       Treat model_path as a HuggingFace model ID instead of a local path"
      echo "  --local_model       Treat model_path as a local path"
      echo "  --resume_run|-r     Name of a previous run to resume from"
      echo "  --help|-h           Show this help message"
      exit 0
      ;;
    --model_path|-m)
      MODEL_PATH="$2"
      # Better detection logic for the new path
      if [[ "$MODEL_PATH" =~ ^[./] || "$MODEL_PATH" =~ ^/ || 
            $(echo "$MODEL_PATH" | tr -cd '/' | wc -c) -gt 1 || 
            "$MODEL_PATH" =~ ^trained_models ]]; then
        # Likely a local path
        IS_MODEL_ID=""
      else
        # Likely a model ID (org/model format with one slash)
        IS_MODEL_ID="--is_model_id"
      fi
      shift 2
      ;;
    --dataset_path|-d)
      DATASET_PATH="$2"
      shift 2
      ;;
    --attribution_method|-a)
      ATTRIBUTION_METHOD="$2"
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
    --output_dir|-o)
      OUTPUT_DIR="--output_dir $2"
      shift 2
      ;;
    --wandb|-w)
      USE_WANDB=true 
      shift
      ;;
    --subset|-s)
      SUBSET="--subset $2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --ignore_pre_generated)
      IGNORE_PRE_GENERATED=true
      shift
      ;;
    --is_model_id)
      IS_MODEL_ID="--is_model_id"
      shift
      ;;
    --local_model)
      IS_MODEL_ID=""  # Force treating as local path even if it looks like a model ID
      shift
      ;;
    --resume_run|-r)
      RESUME_RUN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Evaluating model: $MODEL_PATH"
echo "Using dataset: $DATASET_PATH"
echo "Attribution method: $ATTRIBUTION_METHOD"
echo "WandB logging: $([ "$USE_WANDB" = true ] && echo "enabled" || echo "disabled")"
echo "Model type: $([ -n "$IS_MODEL_ID" ] && echo "HuggingFace model ID" || echo "Local model path")"
echo "Base seed: $SEED"
echo "Pre-generated responses: $([ "$IGNORE_PRE_GENERATED" = true ] && echo "IGNORED" || echo "USED")"
if [[ -n "$RESUME_RUN" ]]; then
  echo "Resuming run: $RESUME_RUN"
fi

# Set WANDB_MODE to offline/online
export WANDB_MODE=online

# Run the evaluation script
python -m src.test_evaluations.eval_trained_dpo \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --attribution_method "$ATTRIBUTION_METHOD" \
  --num_dec_exp "$NUM_DEC_EXP" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --wandb "$USE_WANDB" \
  $([ "$IGNORE_PRE_GENERATED" = true ] && echo "--ignore_pre_generated") \
  $([ -n "$RESUME_RUN" ] && echo "--resume_run $RESUME_RUN") \
  $SUBSET $OUTPUT_DIR $IS_MODEL_ID

echo "Evaluation completed!"