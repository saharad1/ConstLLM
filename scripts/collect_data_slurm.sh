#!/bin/bash
#
# SLURM Wrapper Script for ConstLLM Data Collection
# This script handles SLURM-specific setup and calls the core data collection script
#

# SLURM directives
#SBATCH --job-name=collect_data
#SBATCH --output=logs/collect_data_%A_%a.out
#SBATCH --error=logs/collect_data_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --qos=basic
# SBATCH --array=0-4  # Comment this line to disable array


# Exit on error
set -e

# Display help message
function show_help {
    echo "Usage: sbatch $0 [options]"
    echo ""
    echo "This script submits the data collection job to SLURM."
    echo "All options are passed through to the core data collection script."
    echo ""
    echo "Example:"
    echo "  sbatch $0 --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset ecqa --wandb"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        *)
            # Pass all other arguments to the core script
            CORE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the core script with all arguments
echo "Submitting job to SLURM..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running core script with arguments: ${CORE_ARGS[@]}"
echo "------------------------------------"

"$SCRIPT_DIR/collect_data_core.sh" "${CORE_ARGS[@]}" 