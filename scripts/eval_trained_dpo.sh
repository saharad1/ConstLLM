#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES="0"

# Default paths - adjust these as needed
# MODEL_PATH="models/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_250421_121737_lr2.79e-06_beta6.03/best_model"
MODEL_PATH="unsloth/Qwen2.5-7B-Instruct"
DATASET_PATH="data/collection_data/ecqa/unsloth_Qwen2.5-7B-Instruct/ecqa_20250405_155841_LIME_Qwen2.5/test_1089.jsonl"


# Better detection of model ID vs local path
# Check if it starts with typical absolute/relative path indicators
# OR if it has more than one slash (typical for local paths)
# OR if it starts with "trained_models" (which is clearly a local path in this project)
if [[ "$MODEL_PATH" =~ ^[./] || "$MODEL_PATH" =~ ^/ || 
      $(echo "$MODEL_PATH" | tr -cd '/' | wc -c) -gt 1 || 
      "$MODEL_PATH" =~ ^trained_models ]]; then
  # Likely a local path
  IS_MODEL_ID=""
else
  # Likely a model ID (org/model format with one slash)
  IS_MODEL_ID="--is_model_id"
fi

# Attribution method and other parameters
ATTRIBUTION_METHOD="LIME"
NUM_DEC_EXP=5
TEMPERATURE=0.7
SUBSET=""
SEED=42

# Default is to disable wandb (set to false)
USE_WANDB=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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
    --is_model_id)
      IS_MODEL_ID="--is_model_id"
      shift
      ;;
    --local_model)
      IS_MODEL_ID=""  # Force treating as local path even if it looks like a model ID
      shift
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

# Set WANDB_MODE to offline/online
export WANDB_MODE=offline

# Run the evaluation script
python -m src.test_evaluations.eval_trained_dpo \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --attribution_method "$ATTRIBUTION_METHOD" \
  --num_dec_exp "$NUM_DEC_EXP" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --wandb "$USE_WANDB" \
  $SUBSET $OUTPUT_DIR $IS_MODEL_ID

echo "Evaluation completed!"