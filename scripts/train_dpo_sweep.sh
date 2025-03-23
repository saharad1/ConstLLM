#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES="1"  # This is also set in the Python script, but including here for clarity

# Path to the dataset
DATASET_PATH="dpo_datasets/cleaned_ecqa_dpo_datasets/cleaned_ecqa_250221_181714_LIME"

# Run the DPO training sweep script with command-line arguments
python -m src.pipeline_dpo.train_dpo_unsloth_sweep \
  --dataset_path "$DATASET_PATH" \
  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset_name "ecqa" \
  --similarity_metric "cosine" \
  --diff_threshold 0.2 \
  --sweep_count 10

echo "DPO training sweep completed!"