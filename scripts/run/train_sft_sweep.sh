#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0

# Model ids (examples):
# unsloth/mistral-7b-instruct-v0.3
# unsloth/Meta-Llama-3.1-8B-Instruct
# unsloth/Qwen2.5-7B-Instruct
# unsloth/Llama-3.2-3B-Instruct
# meta-llama/Meta-Llama-3.1-8B-Instruct
# mistralai/Mistral-7B-Instruct-v0.3
# meta-llama/Llama-3.2-3B-Instruct

# Path to the dataset (same layout as DPO: train.jsonl or train_*.jsonl, eval.jsonl)
DATASET_PATH="data/collection_data/ecqa/unsloth_Meta-Llama-3.1-8B-Instruct/ecqa_20250404_120218_LIME_llama3.1"

# Run the SFT training sweep
python -m src.pipeline_sft.train_sft_unsloth_sweep \
  --dataset_path "$DATASET_PATH" \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset_name "ecqa" \
  --similarity_metric "cosine" \
  --diff_threshold_train 0 \
  --diff_threshold_eval 0 \
  --sweep_count 10

echo "SFT training sweep completed!"
