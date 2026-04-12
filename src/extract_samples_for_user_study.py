#!/usr/bin/env python3

import json
import os
import random
from pathlib import Path
from collections import defaultdict

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

def create_user_study_samples():
    """Create user study file with 5 samples from each dataset/model combination, ensuring unique scenario IDs globally"""
    
    # Find all evaluation files - search both directories
    eval_files_collection = find_evaluation_files('/raid/saharad/ConstLLM/data/collection_data')
    eval_files_results = find_evaluation_files('/raid/saharad/ConstLLM/data/eval_results')
    
    # Merge the results
    eval_files = defaultdict(list)
    for key, files in eval_files_collection.items():
        eval_files[key].extend(files)
    for key, files in eval_files_results.items():
        eval_files[key].extend(files)
    
    print(f"Found {len(eval_files)} dataset/model combinations:")
    for (dataset, model), files in eval_files.items():
        print(f"  {dataset} with {model}: {len(files)} files")
    
    user_study_samples = []
    
    # Track used scenario IDs GLOBALLY (not per dataset) to ensure complete uniqueness
    used_scenario_ids = set()
    
    # Collect all samples first and sort by dataset/model for consistent processing
    all_combinations = []
    for (dataset, model), files in eval_files.items():
        all_samples = []
        for file_path in files:
            samples = process_eval_file(file_path)
            all_samples.extend(samples)
        
        if all_samples:
            all_combinations.append((dataset, model, all_samples))
    
    # Sort combinations for consistent ordering
    all_combinations.sort(key=lambda x: (x[0], x[1]))  # Sort by dataset, then model
    
    # Process each dataset/model combination
    for dataset, model, all_samples in all_combinations:
        print(f"\nProcessing {dataset} with {model}...")
        print(f"  Found {len(all_samples)} total samples")
        
        # Filter out already used scenario IDs
        available_samples = [
            sample for sample in all_samples 
            if sample['scenario_id'] not in used_scenario_ids
        ]
        
        print(f"  Found {len(available_samples)} globally unique scenario samples")
        
        if len(available_samples) >= 5:
            # Sort by scenario_id for consistent selection
            available_samples.sort(key=lambda x: x['scenario_id'])
            # Take first 5 unique samples
            selected_samples = available_samples[:5]
        else:
            # Use all available unique samples if less than 5
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
        
        print(f"  Selected {len(selected_samples)} unique samples")
        if selected_samples:
            scenario_ids = [s['scenario_id'] for s in selected_samples]
            print(f"  Scenario IDs: {scenario_ids}")
    
    return user_study_samples

def save_user_study_file(samples, output_file):
    """Save samples to a formatted file for user study"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# User Study: Evaluation Dataset Samples\n\n")
        f.write("This file contains samples from each evaluation dataset, organized by dataset and model.\n")
        f.write("Each scenario ID appears only once globally to ensure no duplicates.\n")
        f.write("Explanations shown are the median-scoring explanations.\n")
        f.write(f"Total samples: {len(samples)}\n\n")
        
        # Group samples by dataset and model
        grouped = defaultdict(list)
        for sample in samples:
            grouped[(sample['dataset'], sample['model'])].append(sample)
        
        f.write("## Table of Contents\n\n")
        for i, (dataset, model) in enumerate(sorted(grouped.keys()), 1):
            f.write(f"{i}. [{dataset} - {model}](#{dataset.lower()}-{model.lower().replace('.', '').replace('-', '')})\n")
        f.write("\n")
        
        # Write detailed samples
        for (dataset, model), dataset_samples in sorted(grouped.items()):
            f.write(f"## {dataset} - {model}\n\n")
            
            for sample in dataset_samples:
                f.write(f"### Sample {sample['sample_number']}\n\n")
                f.write(f"**Scenario ID:** {sample['scenario_id']}\n\n")
                f.write("**Scenario:**\n")
                f.write(f"{sample['scenario']}\n\n")
                f.write("**Model Decision:**\n")
                f.write(f"{sample['decision']}\n\n")
                f.write(f"**Correct Label:** {sample['correct_label']}\n\n")
                f.write("**Median Explanation:**\n")
                f.write(f"{sample['explanation_median']}\n\n")
                f.write(f"**Median Spearman Score:** {sample['median_spearman_score']:.4f}\n\n")
                f.write("---\n\n")
    
    print(f"User study file saved to: {output_file}")

def main():
    # Extract samples
    print("Extracting samples for user study...")
    samples = create_user_study_samples()
    
    # Save to file
    output_file = '/raid/saharad/ConstLLM/user_study_samples.md'
    save_user_study_file(samples, output_file)
    
    # Print summary
    datasets = set(sample['dataset'] for sample in samples)
    models = set(sample['model'] for sample in samples)
    all_scenario_ids = [sample['scenario_id'] for sample in samples]
    
    print(f"\nSummary:")
    print(f"Total samples: {len(samples)}")
    print(f"Unique scenario IDs: {len(set(all_scenario_ids))}")
    print(f"Datasets: {sorted(datasets)}")
    print(f"Models: {sorted(models)}")
    
    # Verify no duplicates
    if len(all_scenario_ids) == len(set(all_scenario_ids)):
        print("✅ All scenario IDs are unique!")
    else:
        print("❌ Warning: Duplicate scenario IDs found!")

if __name__ == "__main__":
    main()