#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=1

# Model ids:
# unsloth/mistral-7b-instruct-v0.3
# unsloth/Meta-Llama-3.1-8B-Instruct
# unsloth/Qwen2.5-7B-Instruct
# unsloth/Llama-3.2-3B-Instruct
# For LIG:
# meta-llama/Meta-Llama-3.1-8B-Instruct
# mistralai/Mistral-7B-Instruct-v0.3
# meta-llama/Llama-3.2-3B-Instruct

# Path to the dataset
DATASET_PATH="data/collection_data/arc_easy/unsloth_Llama-3.2-3B-Instruct/arc_easy_20250417_124056_LIME_llama3.2"

# Run the DPO training sweep script with command-line arguments
python -m src.pipeline_dpo.train_dpo_unsloth_sweep \
  --dataset_path "$DATASET_PATH" \
  --model_id "unsloth/Llama-3.2-3B-Instruct" \
  --dataset_name "arc_easy" \
  --similarity_metric "spearman" \
  --diff_threshold_train 0 \
  --diff_threshold_eval 0 \
  --sweep_count 30 \
  --include_scores \
  --score_scale_factor 10

echo "DPO training sweep completed!"