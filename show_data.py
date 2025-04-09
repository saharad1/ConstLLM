import json
import sys
from datetime import datetime
from pathlib import Path

from src.collect_data.comp_similarity_scores import calculate_cosine_similarity

LOG_DIR = Path("show_logs")
LOG_DIR.mkdir(exist_ok=True)  # Ensure the log directory exists


def setup_logger(file_path):
    """Set up a log file based on the input file name."""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_file_name = f"log_{Path(file_path).stem}_{timestamp}.txt"
    return LOG_DIR / log_file_name


def log_message(message, log_file):
    """Write a message to both console and log file."""
    print(message)  # Print to console
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def load_jsonl(file_path):
    """Load JSONL file and return a list of JSON objects."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def print_scenario(scenario, log_file, num_top_tokens=10):
    """Print and log the scenario details."""
    log_message("=" * 80, log_file)
    log_message(f"Scenario ID: {scenario['scenario_id']}", log_file)
    log_message(f"Decision Prompt:\n{scenario['decision_prompt']}", log_file)
    log_message(f"Model Decision: {scenario['decision_output']}", log_file)
    log_message(f"Correct Label: {scenario['correct_label']}", log_file)
    log_message(f"Explanation Prompt:\n{scenario['explanation_prompt']}", log_file)

    log_message("-" * 80, log_file)
    log_message("Decision Attributions:", log_file)
    log_message(str(scenario["decision_attributions"]), log_file)
    best_exp_top_10_tokens = sorted(scenario["decision_attributions"], key=lambda x: x[1], reverse=True)[:num_top_tokens]
    log_message("Top tokens with highest attributions:", log_file)
    for token, attribution in best_exp_top_10_tokens:
        log_message(f"{token}: {attribution}", log_file)

    spearman_scores_exp = []
    for idx, explanation_attribution in enumerate(scenario["explanation_attributions"]):
        spearman_score_temp = calculate_cosine_similarity(scenario["decision_attributions"], explanation_attribution)
        explanation_scenario = scenario["explanation_outputs"][idx]
        spearman_scores_exp.append((explanation_scenario, explanation_attribution, spearman_score_temp))

    best_explanation = max(spearman_scores_exp, key=lambda x: x[2])
    worst_explanation = min(spearman_scores_exp, key=lambda x: x[2])

    log_message("-" * 80, log_file)
    log_message("Best Explanation:", log_file)
    log_message(str(best_explanation[0]), log_file)
    log_message(f"Best Explanation Spearman: {best_explanation[2]:.3f}", log_file)
    log_message("Best Explanation Attributions:", log_file)
    log_message(str(best_explanation[1]), log_file)

    # Sort and log the top 10 tokens for best explanation
    best_exp_top_10_tokens = sorted(best_explanation[1], key=lambda x: x[1], reverse=True)[:num_top_tokens]
    log_message("Top tokens with highest attributions:", log_file)
    for token, attribution in best_exp_top_10_tokens:
        log_message(f"{token}: {attribution}", log_file)

    log_message("-" * 80, log_file)
    log_message("Worst Explanation:", log_file)
    log_message(str(worst_explanation[0]), log_file)
    log_message(f"Worst Explanation Spearman: {worst_explanation[2]:.3f}", log_file)
    log_message("Worst Explanation Attributions:", log_file)
    log_message(str(worst_explanation[1]), log_file)

    # Sort and log the top 10 tokens for worst explanation
    worst_exp_top_10_tokens = sorted(worst_explanation[1], key=lambda x: x[1], reverse=True)[:num_top_tokens]
    log_message("Top tokens with highest attributions:", log_file)
    for token, attribution in worst_exp_top_10_tokens:
        log_message(f"{token}: {attribution}", log_file)

    log_message("=" * 80 + "\n", log_file)


def run_print_scenarios(file_path, subset=15, num_top_tokens=15):
    log_file = setup_logger(file_path)
    scenarios = load_jsonl(file_path)[:subset]

    for scenario in scenarios:
        print_scenario(scenario, log_file, num_top_tokens)


if __name__ == "__main__":
    file_path = Path(
        "data/collection_data/ecqa/unsloth_Meta-Llama-3.1-8B-Instruct/ecqa_20250404_120218_LIME_llama3.1/ecqa_20250404_120218_LIME_llama3.1_fixed.jsonl"
    )
    run_print_scenarios(str(file_path))
