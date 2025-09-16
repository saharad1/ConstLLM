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

# Create temporary Python script with parameters
TEMP_SCRIPT="temp_extract_$(date +%s).py"

cat > "$TEMP_SCRIPT" << EOF
#!/usr/bin/env python3

import json
import os
import random
import csv
from pathlib import Path
from collections import defaultdict

# Configuration from bash script
TARGET_DATASETS = "${DATASETS}".split(",") if "${DATASETS}" else []
TARGET_FILES = "${FILES}".split() if "${FILES}" else []
NUM_SAMPLES = ${NUM_SAMPLES}
OUTPUT_FILE = "${OUTPUT_FILE}"
CSV_FILE = "${CSV_FILE}"
AUTO_DISCOVER = $([[ "${AUTO_DISCOVER}" == "true" ]] && echo "True" || echo "False")

# Remove empty strings from lists
TARGET_DATASETS = [d.strip() for d in TARGET_DATASETS if d.strip()]
TARGET_FILES = [f.strip() for f in TARGET_FILES if f.strip()]

def extract_sample_data(sample):
    """Extract key information from a sample with median explanation"""
    # Handle both formats - collection_data format and eval_results format
    if 'spearman_best' in sample:
        # collection_data format - need to find median from multiple explanations
        # This format should have explanation_outputs and spearman_scores
        if 'explanation_outputs' in sample and 'spearman_scores' in sample:
            explanations = sample['explanation_outputs']
            spearman_scores = sample['spearman_scores']
            
            # Find median explanation
            sorted_pairs = sorted(zip(spearman_scores, explanations))
            median_idx = len(sorted_pairs) // 2
            median_score, median_explanation = sorted_pairs[median_idx]
            
            return {
                'scenario': sample['decision_prompt'],
                'decision': sample['decision_output'], 
                'correct_label': sample['correct_label'],
                'scenario_id': sample['scenario_id'],
                'explanation_median': median_explanation,
                'median_spearman_score': median_score
            }
        else:
            # Fallback to best explanation if median not available
            return {
                'scenario': sample['decision_prompt'],
                'decision': sample['decision_output'], 
                'correct_label': sample['correct_label'],
                'scenario_id': sample['scenario_id'],
                'explanation_median': sample['spearman_best']['explanation_output'],
                'median_spearman_score': sample['spearman_best'].get('spearman_score', 0.0)
            }
    else:
        # eval_results format - pick median explanation from the list
        explanations = sample['explanation_outputs']
        spearman_scores = sample['spearman_scores']
        
        # Find median explanation
        sorted_pairs = sorted(zip(spearman_scores, explanations))
        median_idx = len(sorted_pairs) // 2
        median_score, median_explanation = sorted_pairs[median_idx]
        
        return {
            'scenario': sample['decision_prompt'],
            'decision': sample['decision_output'], 
            'correct_label': sample['correct_label'],
            'scenario_id': sample['scenario_id'],
            'explanation_median': median_explanation,
            'median_spearman_score': median_score
        }

def process_eval_file(file_path):
    """Process a single evaluation file and return all samples"""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        samples.append(extract_sample_data(sample))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {file_path}: {e}")
                        continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return samples

def find_evaluation_files(base_dir):
    """Find all evaluation files and categorize by dataset and model"""
    results = defaultdict(list)
    
    base_path = Path(base_dir)
    
    # Handle both collection_data and eval_results directories
    for eval_file in base_path.rglob("*results.jsonl"):
        path_parts = eval_file.parts
        
        if 'eval_results' in path_parts:
            # Handle eval_results format
            idx = path_parts.index('eval_results')
            if idx + 2 < len(path_parts):
                dataset = path_parts[idx + 1]  # e.g., arc_challenge
                
                # Extract model info - could be in different positions
                model_info = path_parts[idx + 2]  # e.g., huggingface, Llama-3.2-3B-Instruct, etc.
                
                if model_info == 'huggingface' and idx + 3 < len(path_parts):
                    # Format: eval_results/dataset/huggingface/model_name/
                    model = path_parts[idx + 3]
                elif model_info in ['huggingface']:
                    continue  # Skip this pattern for now
                else:
                    # Format: eval_results/dataset/model_name/ or eval_results/dataset/model_name/training_info/
                    model = model_info
                
                results[(dataset, model)].append(str(eval_file))
        
        elif 'collection_data' in path_parts:
            # Handle collection_data format (original)
            idx = path_parts.index('collection_data')
            if idx + 1 < len(path_parts):
                dataset = path_parts[idx + 1]
                
                # Extract model name
                if idx + 2 < len(path_parts):
                    model_dir = path_parts[idx + 2]
                    # Extract model name from directory (e.g., "unsloth_Llama-3.2-3B-Instruct" -> "Llama-3.2-3B-Instruct")
                    if model_dir.startswith('unsloth_'):
                        model = model_dir[8:]  # Remove "unsloth_" prefix
                    elif model_dir.startswith('meta-llama_'):
                        model = model_dir[11:]  # Remove "meta-llama_" prefix
                    else:
                        model = model_dir
                        
                    results[(dataset, model)].append(str(eval_file))
    
    return results

def extract_dataset_model_from_path(file_path):
    """Extract dataset, model, attribution method, and training type from file path"""
    path_parts = Path(file_path).parts
    filename = Path(file_path).name

    if 'eval_results' in path_parts:
        # Handle eval_results format
        idx = path_parts.index('eval_results')
        if idx + 2 < len(path_parts):
            dataset = path_parts[idx + 1]  # e.g., arc_easy, ecqa

            # Extract model info - could be in different positions
            model_info = path_parts[idx + 2]  # e.g., huggingface, Llama-3.2-3B-Instruct, etc.

            if model_info == 'huggingface' and idx + 3 < len(path_parts):
                # Format: eval_results/dataset/huggingface/model_name/
                model = path_parts[idx + 3]
                training_type = 'huggingface'
            else:
                # Format: eval_results/dataset/model_name/ or eval_results/dataset/model_name/training_info/
                model = model_info
                training_type = 'fine-tuned'

            # Extract attribution method from filename
            if 'LIME' in filename:
                attribution_method = 'LIME'
            elif 'LIG' in filename:
                attribution_method = 'LIG'
            else:
                attribution_method = 'unknown'

            return dataset, model, attribution_method, training_type
    
    elif 'collection_data' in path_parts:
        # Handle collection_data format
        idx = path_parts.index('collection_data')
        if idx + 1 < len(path_parts):
            dataset = path_parts[idx + 1]

            # Extract model name
            if idx + 2 < len(path_parts):
                model_dir = path_parts[idx + 2]
                # Extract model name from directory (e.g., "unsloth_Llama-3.2-3B-Instruct" -> "Llama-3.2-3B-Instruct")
                if model_dir.startswith('unsloth_'):
                    model = model_dir[8:]  # Remove "unsloth_" prefix
                elif model_dir.startswith('meta-llama_'):
                    model = model_dir[11:]  # Remove "meta-llama_" prefix
                else:
                    model = model_dir

                # Extract attribution method from filename
                if 'LIME' in filename:
                    attribution_method = 'LIME'
                elif 'LIG' in filename:
                    attribution_method = 'LIG'
                else:
                    attribution_method = 'unknown'

                return dataset, model, attribution_method, 'collection_data'

    # Fallback: try to extract from filename
    print(f"Warning: Could not extract dataset/model from path structure for {file_path}")
    return "unknown", "unknown", "unknown", "unknown"

def create_user_study_samples_from_datasets():
    """Create user study file from specific datasets or files"""

    if TARGET_FILES:
        # Use specific files provided by user - treat each file individually
        eval_files = []
        for file_path in TARGET_FILES:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            dataset, model, attribution_method, training_type = extract_dataset_model_from_path(file_path)
            eval_files.append((dataset, model, attribution_method, training_type, file_path))
    elif AUTO_DISCOVER:
        # Auto-discover files
        eval_files_collection = find_evaluation_files('/raid/saharad/ConstLLM/data/collection_data')
        eval_files_results = find_evaluation_files('/raid/saharad/ConstLLM/data/eval_results')

        # Merge the results
        eval_files = defaultdict(list)
        for key, files in eval_files_collection.items():
            eval_files[key].extend(files)
        for key, files in eval_files_results.items():
            eval_files[key].extend(files)
    else:
        # Filter by specified datasets
        eval_files_collection = find_evaluation_files('/raid/saharad/ConstLLM/data/collection_data')
        eval_files_results = find_evaluation_files('/raid/saharad/ConstLLM/data/eval_results')

        # Merge the results and filter by target datasets
        all_eval_files = defaultdict(list)
        for key, files in eval_files_collection.items():
            all_eval_files[key].extend(files)
        for key, files in eval_files_results.items():
            all_eval_files[key].extend(files)

        # Filter by target datasets
        eval_files = defaultdict(list)
        for (dataset, model), files in all_eval_files.items():
            if dataset in TARGET_DATASETS:
                eval_files[(dataset, model)] = files
    
    if not eval_files:
        print("No evaluation files found!")
        return []

    if TARGET_FILES:
        # For individual files, show each file
        print(f"Found {len(eval_files)} individual files:")
        for dataset, model, attribution_method, training_type, file_path in eval_files:
            print(f"  {dataset} | {model} | {attribution_method} | {training_type}: {file_path}")
    else:
        # For grouped files, show combinations
        print(f"Found {len(eval_files)} dataset/model combinations:")
        for (dataset, model), files in eval_files.items():
            print(f"  {dataset} with {model}: {len(files)} files")
            for f in files:
                print(f"    - {f}")

    user_study_samples = []

    # Track used scenario IDs GLOBALLY to ensure complete uniqueness across all datasets/models
    used_scenario_ids = set()

    if TARGET_FILES:
        # Process each file individually
        for i, (dataset, model, attribution_method, training_type, file_path) in enumerate(eval_files, 1):
            print(f"  Processing file {i}/{len(eval_files)}: {dataset} | {model} | {attribution_method} | {training_type}")
            print(f"    File: {file_path}")

            samples = process_eval_file(file_path)
            # Add file path to each sample
            for sample in samples:
                sample['file_path'] = file_path

            print(f"    Found {len(samples)} total samples")

            # Filter out already used scenario IDs (globally unique)
            available_samples = [
                sample for sample in samples
                if sample['scenario_id'] not in used_scenario_ids
            ]

            print(f"    Found {len(available_samples)} globally unique scenario samples")

            if len(available_samples) >= NUM_SAMPLES:
                # Sort by scenario_id for consistent selection
                available_samples.sort(key=lambda x: x['scenario_id'])
                # Take first N unique samples
                selected_samples = available_samples[:NUM_SAMPLES]
            else:
                # Use all available unique samples if less than N
                selected_samples = available_samples

            # Mark these scenario IDs as used GLOBALLY
            for sample in selected_samples:
                used_scenario_ids.add(sample['scenario_id'])

            # Add metadata and store
            for j, sample in enumerate(selected_samples, 1):
                user_study_sample = {
                    'dataset': dataset,
                    'model': model,
                    'attribution_method': attribution_method,
                    'training_type': training_type,
                    'file_number': i,
                    'sample_number': j,
                    **sample
                }
                user_study_samples.append(user_study_sample)

            print(f"    Selected {len(selected_samples)} unique samples")
            if selected_samples:
                scenario_ids = [s['scenario_id'] for s in selected_samples]
                print(f"    Scenario IDs: {scenario_ids}")
    else:
        # Process grouped files (original logic for datasets/auto-discover)
        # Collect all combinations and sort for consistent processing
        all_combinations = []
        for (dataset, model), files in eval_files.items():
            all_combinations.append((dataset, model, files))

        # Sort combinations for consistent ordering
        all_combinations.sort(key=lambda x: (x[0], x[1]))  # Sort by dataset, then model

        # Process each dataset/model combination
        for dataset, model, files in all_combinations:
                print(f"  Processing {dataset} with {model}...")

                all_samples = []
                for file_path in files:
                    samples = process_eval_file(file_path)
                    # Add file path to each sample
                    for sample in samples:
                        sample['file_path'] = file_path
                    all_samples.extend(samples)

                print(f"    Found {len(all_samples)} total samples")

                # Filter out already used scenario IDs (globally unique)
                available_samples = [
                    sample for sample in all_samples
                    if sample['scenario_id'] not in used_scenario_ids
                ]

                print(f"    Found {len(available_samples)} globally unique scenario samples")

                if len(available_samples) >= NUM_SAMPLES:
                    # Sort by scenario_id for consistent selection
                    available_samples.sort(key=lambda x: x['scenario_id'])
                    # Take first N unique samples
                    selected_samples = available_samples[:NUM_SAMPLES]
                else:
                    # Use all available unique samples if less than N
                    selected_samples = available_samples

                # Mark these scenario IDs as used GLOBALLY
                for sample in selected_samples:
                    used_scenario_ids.add(sample['scenario_id'])

                # Add metadata and store
                for i, sample in enumerate(selected_samples, 1):
                    user_study_sample = {
                        'dataset': dataset,
                        'model': model,
                        'sample_number': i,
                        **sample
                    }
                    user_study_samples.append(user_study_sample)

                print(f"    Selected {len(selected_samples)} unique samples")
                if selected_samples:
                    scenario_ids = [s['scenario_id'] for s in selected_samples]
                    print(f"    Scenario IDs: {scenario_ids}")
    
    return user_study_samples

def save_user_study_file(samples, output_file):
    """Save samples to a formatted file for user study"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# User Study: Evaluation Dataset Samples\\n\\n")
        f.write("This file contains samples from each evaluation dataset, organized by dataset and model.\\n")
        f.write("Each scenario ID appears only once globally to ensure no duplicates.\\n")
        f.write("Explanations shown are the median-scoring explanations.\\n")
        f.write(f"Total samples: {len(samples)}\\n\\n")
        
        # Group samples by dataset, model, attribution method, and training type
        grouped = defaultdict(list)
        for sample in samples:
            key = (sample['dataset'], sample['model'], sample.get('attribution_method', 'unknown'), sample.get('training_type', 'unknown'))
            grouped[key].append(sample)

        f.write("## Table of Contents\\n\\n")
        for i, (dataset, model, attribution_method, training_type) in enumerate(sorted(grouped.keys()), 1):
            anchor = f"{dataset.lower()}-{model.lower().replace('.', '').replace('-', '')}-{attribution_method.lower()}-{training_type.lower().replace('-', '')}"
            f.write(f"{i}. [{dataset} - {model} - {attribution_method} - {training_type}](#{anchor})\\n")
        f.write("\\n")

        # Write detailed samples
        for (dataset, model, attribution_method, training_type), dataset_samples in sorted(grouped.items()):
            f.write(f"## {dataset} - {model} - {attribution_method} - {training_type}\\n\\n")
            
            for sample in dataset_samples:
                f.write(f"### Sample {sample['sample_number']}\\n\\n")
                f.write(f"**Scenario ID:** {sample['scenario_id']}\\n\\n")
                f.write("**Scenario:**\\n")
                f.write(f"{sample['scenario']}\\n\\n")
                f.write("**Model Decision:**\\n")
                f.write(f"{sample['decision']}\\n\\n")
                f.write(f"**Correct Label:** {sample['correct_label']}\\n\\n")
                f.write("**Median Explanation:**\\n")
                f.write(f"{sample['explanation_median']}\\n\\n")
                f.write(f"**Median Spearman Score:** {sample['median_spearman_score']:.4f}\\n\\n")
                f.write("---\\n\\n")
    
    print(f"User study file saved to: {output_file}")

def save_user_study_csv(samples, csv_file):
    """Save samples to a CSV file for user study"""

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'ScenarioID', 'AttributionMethod', 'Dataset', 'Model', 'TrainingType', 'Question', 'ModelAnswer',
            'CorrectLabel', 'Explanation', 'SpearmanScore', 'DatasetPath', 'ModelPath'
        ])

        # Write data rows
        for sample in samples:
            # Extract dataset and model paths from file path
            file_path = sample.get('file_path', '')
            dataset_path = f"/raid/saharad/ConstLLM/data/*/{sample['dataset']}"
            model_path = file_path if file_path else f"*/{sample['model']}/*"

            writer.writerow([
                sample['scenario_id'],
                sample.get('attribution_method', 'unknown'),
                sample['dataset'],
                sample['model'],
                sample.get('training_type', 'unknown'),
                sample['scenario'].replace('\\n', ' ').replace('\\r', ' '),  # Clean newlines
                sample['decision'].replace('\\n', ' ').replace('\\r', ' '),
                sample['correct_label'],
                sample['explanation_median'].replace('\\n', ' ').replace('\\r', ' '),
                f"{sample['median_spearman_score']:.4f}",
                dataset_path,
                model_path
            ])

    print(f"User study CSV saved to: {csv_file}")

if __name__ == "__main__":
    # Extract samples
    print("Extracting samples for user study...")
    samples = create_user_study_samples_from_datasets()
    
    if not samples:
        print("No samples extracted. Exiting.")
        exit(1)
    
    # Save to files
    save_user_study_file(samples, OUTPUT_FILE)
    save_user_study_csv(samples, CSV_FILE)

    # Print summary
    datasets = set(sample['dataset'] for sample in samples)
    models = set(sample['model'] for sample in samples)

    print(f"\\nSummary:")
    print(f"Total samples: {len(samples)}")
    print(f"Datasets: {sorted(datasets)}")
    print(f"Models: {sorted(models)}")
    print(f"Markdown file: {OUTPUT_FILE}")
    print(f"CSV file: {CSV_FILE}")
EOF

# Run the temporary Python script
echo ""
echo "Running extraction..."
python "$TEMP_SCRIPT"

# Clean up
rm "$TEMP_SCRIPT"

echo ""
echo "✅ User study extraction completed!"
echo "📁 Markdown file: $OUTPUT_FILE"
echo "📄 CSV file: $CSV_FILE"