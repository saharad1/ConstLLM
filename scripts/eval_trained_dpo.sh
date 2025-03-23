#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES="1"  # Use GPU 1 for evaluation

# Default paths - adjust these as needed
MODEL_PATH="trained_models/ecqa_models/LLama-instruct-8b/ecqa_250320_170948/final-model"
DATASET_PATH="data/test_datasets/ecqa_test.jsonl"
OUTPUT_DIR="data/eval_results"

# Attribution method and other parameters
ATTRIBUTION_METHOD="LIME"
NUM_DEC_EXP=5
TEMPERATURE=0.7

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --attribution_method)
      ATTRIBUTION_METHOD="$2"
      shift 2
      ;;
    --num_dec_exp)
      NUM_DEC_EXP="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --no_wandb)
      NO_WANDB="--no_wandb"
      shift
      ;;
    --subset)
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

# Run the evaluation script
python -m src.test_evaluations.eval_trained_dpo \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --attribution_method "$ATTRIBUTION_METHOD" \
  --num_dec_exp "$NUM_DEC_EXP" \
  --temperature "$TEMPERATURE" \
  --output_dir "$OUTPUT_DIR" \
  $NO_WANDB $SUBSET

echo "Evaluation completed!"