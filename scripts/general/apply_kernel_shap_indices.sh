#!/bin/bash
#
# Apply kernel SHAP selected indices to dataset test sets
# - Reads indices from data/dataset_splits/kernel_shap_indices.json by default
# - Scans data/collection_data for test_*.jsonl files per dataset
# - Matches test file by size and filters by scenario_id
# - Writes outputs under data/kernel_shap_datasets mirroring directory structure
#

# Exit on error
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ConstLLM

# Defaults
INDICES_FILE="data/dataset_splits/kernel_shap_indices.json"
COLLECTION_ROOT="data/collection_data"
OUTPUT_ROOT="data/kernel_shap_datasets"
DATASETS=(arc_easy arc_challenge ecqa codah)

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --indices_file FILE     Selected indices JSON (default: $INDICES_FILE)"
    echo "  --collection_root DIR   Root of collected datasets (default: $COLLECTION_ROOT)"
    echo "  --output_root DIR       Root for filtered outputs (default: $OUTPUT_ROOT)"
    echo "  --datasets NAMES        Space-separated datasets (default: ${DATASETS[*]})"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --indices_file data/dataset_splits/kernel_shap_indices.json \\
          --collection_root data/collection_data \\
          --output_root data/kernel_shap_datasets \\
          --datasets arc_easy ecqa"
}

# Parse CLI
while [[ $# -gt 0 ]]; do
    case "$1" in
        --indices_file)
            INDICES_FILE="$2"; shift 2;;
        --collection_root)
            COLLECTION_ROOT="$2"; shift 2;;
        --output_root)
            OUTPUT_ROOT="$2"; shift 2;;
        --datasets)
            shift
            DATASETS=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        -h|--help)
            show_help; exit 0;;
        *)
            echo "Unknown option: $1"; show_help; exit 1;;
    esac
done

# Ensure running from project root
if [[ ! -d "src" ]]; then
    echo "Error: run from project root (src/ must exist). Current: $(pwd)"; exit 1
fi

# Validate files/dirs
[[ -f "$INDICES_FILE" ]] || { echo "Missing indices file: $INDICES_FILE"; exit 1; }
[[ -d "$COLLECTION_ROOT" ]] || { echo "Missing collection root: $COLLECTION_ROOT"; exit 1; }
mkdir -p "$OUTPUT_ROOT"

echo "Applying kernel SHAP indices..."
echo "Indices: $INDICES_FILE"
echo "Collection root: $COLLECTION_ROOT"
echo "Output root: $OUTPUT_ROOT"
printf "Datasets: %s\n" "${DATASETS[*]}"

python src/apply_kernel_shap_indices.py \
    --indices_file "$INDICES_FILE" \
    --collection_root "$COLLECTION_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --datasets "${DATASETS[@]}"

echo "Done. Outputs written under: $OUTPUT_ROOT"
