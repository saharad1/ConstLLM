#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES="1"  # Use GPU 1 for evaluation

# Default paths - adjust these as needed
MODEL_PATH="trained_models/ecqa_models/meta-llama/Meta-Llama-3.1-8B-Instruct/ecqa_cosine_lr5.57e-05_beta0.10_250323_193605/final-model"
DATASET_PATH="dpo_datasets/cleaned_ecqa_dpo_datasets/cleaned_ecqa_250221_181714_LIME/test_1089.jsonl"
# OUTPUT_DIR="data/eval_results"


# Attribution method and other parameters
ATTRIBUTION_METHOD="LIME"
NUM_DEC_EXP=5
TEMPERATURE=0.7
SUBSET=""

# Default is to disable wandb (set to false)
USE_WANDB=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path|-m)
      MODEL_PATH="$2"
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
      USE_WANDB=true  # Enable wandb
      shift
      ;;
    --subset|-s)
      SUBSET="--subset $2"
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

# Run the evaluation script
python -m src.test_evaluations.eval_trained_dpo \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --attribution_method "$ATTRIBUTION_METHOD" \
  --num_dec_exp "$NUM_DEC_EXP" \
  --temperature "$TEMPERATURE" \
  --wandb "$USE_WANDB" \
  $SUBSET $OUTPUT_DIR

echo "Evaluation completed!"