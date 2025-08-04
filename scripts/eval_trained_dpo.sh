#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0

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
MODEL_PATH="models/ecqa/Llama-3.2-3B-Instruct/ecqa_250801_124134_lr9.97e-06_beta9.73/best_model"
DATASET_PATH="data/collection_data/ecqa/meta-llama_Llama-3.2-3B-Instruct/ecqa_20250403_133655_LIG_llama3.2/test_1089.jsonl"
OUTPUT_DIR=""

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
export WANDB_MODE=online

# Run the evaluation script with --ignore_pre_generated flag
python -m src.test_evaluations.eval_trained_dpo \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --attribution_method "$ATTRIBUTION_METHOD" \
  --num_dec_exp "$NUM_DEC_EXP" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --wandb "$USE_WANDB" \
  # --ignore_pre_generated \
  $SUBSET $OUTPUT_DIR $IS_MODEL_ID

echo "Evaluation completed!"