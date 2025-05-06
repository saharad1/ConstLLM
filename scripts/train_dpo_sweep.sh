#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES="0"  # This is also set in the Python script, but including here for clarity

# Path to the dataset
DATASET_PATH="data/collection_data/arc_easy/unsloth_Meta-Llama-3.1-8B-Instruct/arc_easy_20250424_152104_LIME_llama3.1"

# Run the DPO training sweep script with command-line arguments
python -m src.pipeline_dpo.train_dpo_unsloth_sweep \
  --dataset_path "$DATASET_PATH" \
  --model_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
  --dataset_name "ecqa" \
  --similarity_metric "cosine" \
  --diff_threshold_train 0 \
  --diff_threshold_eval 0 \
  --sweep_count 10 \
  --include_scores \
  --score_scale_factor 100 \

echo "DPO training sweep completed!"