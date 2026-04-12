#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Auto-detect a working GPU by UUID (bypasses broken GPU enumeration on nodes with faulty GPUs)
# Finds the first A100 GPU's UUID from nvidia-smi and uses it directly
GPU_UUID=$(nvidia-smi -L 2>/dev/null | grep "A100" | head -n1 | sed 's/.*UUID: \(GPU-[^)]*\)).*/\1/')
if [ -n "$GPU_UUID" ]; then
  echo "Detected working GPU: $GPU_UUID"
  export CUDA_VISIBLE_DEVICES="$GPU_UUID"
else
  echo "WARNING: Could not detect A100 GPU UUID, falling back to CUDA_VISIBLE_DEVICES=0"
  export CUDA_VISIBLE_DEVICES=0
fi

# Out of domain datasets:
# codah,llama3.1: data/collection_data/codah/unsloth_Meta-Llama-3.1-8B-Instruct/codah_20250415_125210_LIME_llama3.1/test_278.jsonl
# codah,llama3.2: data/collection_data/codah/unsloth_Llama-3.2-3B-Instruct/codah_20250506_085629_LIME_llama3.2/test_272.jsonl
# arc_challenge,llama3.1: data/collection_data/arc_challenge/unsloth_Meta-Llama-3.1-8B-Instruct/arc_challenge_20250415_155822_LIME_llama3.1/test_260.jsonl
# arc_challenge,llama3.2: data/collection_data/arc_challenge/unsloth_Llama-3.2-3B-Instruct/arc_challenge_20250421_094925_LIME_llama3.2/test_255.jsonl

# Out of domain models (SFT):
# ecqa,llama3.1: models/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_<run_name>/best_model
# ecqa,llama3.2: models/ecqa/Llama-3.2-3B-Instruct/ecqa_<run_name>/best_model
# arc_easy,llama3.1: models/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_<run_name>/best_model
# arc_easy,llama3.2: models/arc_easy/Llama-3.2-3B-Instruct/arc_easy_<run_name>/best_model

# Default paths - adjust these as needed
MODEL_PATH="models/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_260210_193118_lr6.95e-06/best_model"
DATASET_PATH="data/collection_data/ecqa/unsloth_Meta-Llama-3.1-8B-Instruct/ecqa_20250404_120218_LIME_llama3.1/test_1089.jsonl"
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
# KSHAP


# Attribution method and other parameters
ATTRIBUTION_METHOD="LIME"
NUM_DEC_EXP=5
TEMPERATURE=0.7
SUBSET=""
SEED=42
NO_QUANTIZATION=true


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
      echo "  --no_quantization   Disable 4-bit quantization (default: enabled)"
      echo "  --quantization      Enable 4-bit quantization"
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
    --no_quantization)
      NO_QUANTIZATION=true
      shift
      ;;
    --quantization)
      NO_QUANTIZATION=false
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
echo "Pre-generated responses: $([ "$IGNORE_PRE_GENERATED" = true ] && echo "IGNORED" || echo "USED")"
echo "Quantization: $([ "$NO_QUANTIZATION" = true ] && echo "DISABLED" || echo "ENABLED")"
if [[ -n "$RESUME_RUN" ]]; then
  echo "Resuming run: $RESUME_RUN"
fi

# Set WANDB_MODE to offline/online
export WANDB_MODE=online

# Run the evaluation script
python -m src.test_evaluations.eval_trained_sft \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --attribution_method "$ATTRIBUTION_METHOD" \
  --num_dec_exp "$NUM_DEC_EXP" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --wandb "$USE_WANDB" \
  $([ "$IGNORE_PRE_GENERATED" = true ] && echo "--ignore_pre_generated") \
  $([ "$NO_QUANTIZATION" = true ] && echo "--no_quantization") \
  $([ -n "$RESUME_RUN" ] && echo "--resume_run $RESUME_RUN") \
  $SUBSET $OUTPUT_DIR $IS_MODEL_ID

echo "Evaluation completed!"
