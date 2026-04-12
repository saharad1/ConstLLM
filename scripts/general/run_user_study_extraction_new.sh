#!/bin/bash

# User Study Sample Extraction Script
# Usage: ./run_user_study_extraction.sh --datasets "dataset1,dataset2,..." --num-samples N
# Or: ./run_user_study_extraction.sh --auto-discover

set -e

# Default values
DEFAULT_NUM_SAMPLES=5
OUTPUT_DIR="outputs/user_study_extraction"
DEFAULT_OUTPUT_FILE="$OUTPUT_DIR/user_study_samples.md"
DEFAULT_CSV_FILE="$OUTPUT_DIR/user_study_samples.csv"
PYTHON_SCRIPT="src/extract_user_study_samples.py"

# Help function
show_help() {
    cat << EOF
User Study Sample Extraction Script

Usage:
    $0 [OPTIONS]

OPTIONS:
    -d, --datasets DATASETS     Comma-separated list of dataset names (e.g., "arc_challenge,hellaswag")
    -f, --files FILES           Comma-separated list of specific jsonl file paths
    -n, --num-samples N         Number of samples per dataset/model combination (default: $DEFAULT_NUM_SAMPLES)
    -o, --output OUTPUT_FILE    Markdown output file path (default: $DEFAULT_OUTPUT_FILE)
    -c, --csv CSV_FILE          CSV output file path (default: $DEFAULT_CSV_FILE)
    -a, --auto-discover         Auto-discover all evaluation files in standard locations
    -h, --help                  Show this help message

EXAMPLES:
    # Extract 3 samples from specific datasets
    $0 --datasets "arc_challenge,hellaswag,mmlu" --num-samples 3

    # Extract samples from specific evaluation files
    $0 --files "file1.jsonl,file2.jsonl,file3.jsonl" --num-samples 5

    # Auto-discover all evaluation files and extract 5 samples each
    $0 --auto-discover --num-samples 5

    # Specify custom output files
    $0 --datasets "arc_challenge" --output "my_study.md" --csv "my_study.csv"

EVALUATION FILE DISCOVERY:
    The script will automatically find evaluation files for the specified datasets.
    Expected directory structure:
    - /raid/saharad/ConstLLM/data/collection_data/DATASET/MODEL/.../*.jsonl
    - /raid/saharad/ConstLLM/data/eval_results/DATASET/MODEL/.../*results.jsonl

EOF
}

# Parse command line arguments
LIME_ARC_EASY_DATASETS="data/eval_results/arc_easy/huggingface/Meta-Llama-3.1-8B-Instruct/eval_250505_183622_test_521_LIME/eval_250505_183622_test_521_LIME_results.jsonl \
data/eval_results/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_250510_194800_lr4.65e-06_beta5.64/eval_250606_123356_test_521_LIME/eval_250606_123356_test_521_LIME_results.jsonl \
data/eval_results/arc_easy/huggingface/Llama-3.2-3B-Instruct/eval_250505_173301_test_517_LIME/eval_250505_173301_test_517_LIME_results.jsonl \
data/eval_results/arc_easy/Llama-3.2-3B-Instruct/arc_easy_250516_015641_lr6.32e-06_beta8.84/eval_250604_105055_test_517_LIME/eval_250604_105055_test_517_LIME_results.jsonl"
LIME_ECQA_DATASETS="data/eval_results/ecqa/huggingface/Meta-Llama-3.1-8B-Instruct/eval_test_1089_LIME_250414_154947/eval_test_1089_LIME_250414_154947_results.jsonl \
data/eval_results/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_250508_224902_lr4.21e-06_beta5.13/eval_250529_093246_test_1089_LIME/eval_250529_093246_test_1089_LIME_results.jsonl \
data/eval_results/ecqa/huggingface/Llama-3.2-3B-Instruct/eval_250505_103954_test_1089_LIME/eval_250505_103954_test_1089_LIME_results.jsonl \
data/eval_results/ecqa/Llama-3.2-3B-Instruct/ecqa_250513_133025_lr9.55e-06_beta8.44/eval_250603_131140_test_1089_LIME/eval_250603_131140_test_1089_LIME_results.jsonl \
"
LIG_ARC_EASY_DATASETS="data/eval_results/arc_easy/huggingface/Meta-Llama-3.1-8B-Instruct/eval_250824_172702_test_521_LIG_with_pregen/eval_250824_172702_test_521_LIG_with_pregen_results.jsonl \
data/eval_results/arc_easy/Meta-Llama-3.1-8B-Instruct/arc_easy_250806_181728_lr6.86e-06_beta8.41/eval_250829_172353_test_521_LIG_no_pregen/eval_250829_172353_test_521_LIG_no_pregen_results.jsonl \
data/eval_results/arc_easy/huggingface/Llama-3.2-3B-Instruct/eval_250818_090347_test_521_LIG_with_pregen/eval_250818_090347_test_521_LIG_with_pregen_results.jsonl \
data/eval_results/arc_easy/Llama-3.2-3B-Instruct/arc_easy_250805_195416_lr3.84e-06_beta9.20/eval_250821_114049_test_521_LIG_no_pregen/eval_250821_114049_test_521_LIG_no_pregen_results.jsonl \
"
LIG_ECQA_DATASETS="data/eval_results/ecqa/huggingface/Meta-Llama-3.1-8B-Instruct/eval_250829_194757_test_1089_LIG_with_pregen/eval_250829_194757_test_1089_LIG_with_pregen_results.jsonl \
data/eval_results/ecqa/Meta-Llama-3.1-8B-Instruct/ecqa_250818_054203_lr8.96e-06_beta7.04/eval_250825_154304_test_1089_LIG_no_pregen/eval_250825_154304_test_1089_LIG_no_pregen_results.jsonl \
data/eval_results/ecqa/huggingface/Llama-3.2-3B-Instruct/eval_250819_201425_test_1089_LIG_with_pregen/eval_250819_201425_test_1089_LIG_with_pregen_results.jsonl \
data/eval_results/ecqa/Llama-3.2-3B-Instruct/ecqa_250808_050834_lr9.56e-06_beta9.93/eval_250821_120034_test_1089_LIG_no_pregen/eval_250821_120034_test_1089_LIG_no_pregen_results.jsonl \
"

# Combined lists
# ALL_LIME_DATASETS="$LIME_ARC_EASY_DATASETS $LIME_ECQA_DATASETS"
# ALL_LIG_DATASETS="$LIG_ARC_EASY_DATASETS $LIG_ECQA_DATASETS"
# ALL_ARC_EASY_DATASETS="$LIME_ARC_EASY_DATASETS $LIG_ARC_EASY_DATASETS"
# ALL_ECQA_DATASETS="$LIME_ECQA_DATASETS $LIG_ECQA_DATASETS"
ALL_DATASETS="$LIME_ARC_EASY_DATASETS $LIME_ECQA_DATASETS $LIG_ARC_EASY_DATASETS $LIG_ECQA_DATASETS"
FILES="$ALL_DATASETS"
NUM_SAMPLES=$DEFAULT_NUM_SAMPLES
OUTPUT_FILE=$DEFAULT_OUTPUT_FILE
CSV_FILE=$DEFAULT_CSV_FILE
AUTO_DISCOVER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--datasets)
            DATASETS="$2"
            shift 2
            ;;
        -f|--files)
            FILES="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -c|--csv)
            CSV_FILE="$2"
            shift 2
            ;;
        -a|--auto-discover)
            AUTO_DISCOVER=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validation
if ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]] || [ "$NUM_SAMPLES" -le 0 ]; then
    echo "Error: num-samples must be a positive integer, got: $NUM_SAMPLES"
    exit 1
fi

if [[ -z "$DATASETS" ]] && [[ -z "$FILES" ]] && [[ "$AUTO_DISCOVER" != true ]]; then
    echo "Error: No datasets or files specified. Use --datasets, --files, or --auto-discover"
    show_help
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Display configuration
echo "=========================================="
echo "User Study Sample Extraction Configuration"
echo "=========================================="
if [[ "$AUTO_DISCOVER" == true ]]; then
    echo "Datasets: Auto-discovering all evaluation datasets"
elif [[ -n "$FILES" ]]; then
    echo "Files: $FILES"
else
    echo "Datasets: $DATASETS"
fi
echo "Samples per combination: $NUM_SAMPLES"
echo "Markdown output file: $OUTPUT_FILE"
echo "CSV output file: $CSV_FILE"
echo "=========================================="

# Build command line arguments for the Python script
PYTHON_ARGS=()

if [[ -n "$DATASETS" ]]; then
    PYTHON_ARGS+=("--datasets" "$DATASETS")
fi

if [[ -n "$FILES" ]]; then
    PYTHON_ARGS+=("--files" "$FILES")
fi

if [[ "$AUTO_DISCOVER" == true ]]; then
    PYTHON_ARGS+=("--auto-discover")
fi

PYTHON_ARGS+=("--num-samples" "$NUM_SAMPLES")
PYTHON_ARGS+=("--output" "$OUTPUT_FILE")
PYTHON_ARGS+=("--csv" "$CSV_FILE")

# Run the Python script
echo ""
echo "Running extraction..."
python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"

echo ""
echo "✅ User study extraction completed!"
echo "📁 Markdown file: $OUTPUT_FILE"
echo "📄 CSV file: $CSV_FILE"
