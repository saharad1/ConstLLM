import json
import sys
from pathlib import Path

from collect_data.comp_score import compute_spearman_score


def load_jsonl(file_path):
    """Load JSONL file and return a list of JSON objects."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def print_scenario(scenario):
    """Print the scenario details in a nicely formatted way."""
    print("=" * 80)
    print(f"Scenario ID: {scenario['scenario_id']}")
    print(f"Decision Prompt:\n{scenario['decision_prompt']}")
    print(f"Model Decision: {scenario['decision_output']}")
    print(f"Correct Label: {scenario['correct_label']}")
    print(f"Explanation Prompt:\n{scenario['explanation_prompt']}")
    print(f"Best Explanation: {scenario['explanation_best']['explanation_output']}")
    print(
        f"Best Explanation Score: {scenario['explanation_best']['spearman_score']:.3f}"
    )

    print("-" * 80)
    print("Decision Attributions:")
    print(scenario["decision_attributions"])

    spearman_scores_exp = []
    for idx, explanation_attribution in enumerate(scenario["explanation_attributions"]):
        spearman_score_temp = compute_spearman_score(
            scenario["decision_attributions"], explanation_attribution
        )
        explanation_scenario = scenario["explanation_outputs"][idx]
        spearman_scores_exp.append(
            (explanation_scenario, explanation_attribution, spearman_score_temp)
        )

    best_explanation = max(spearman_scores_exp, key=lambda x: x[2])
    worst_explanation = min(spearman_scores_exp, key=lambda x: x[2])

    print("-" * 80)
    print("Explanation Attributions:")
    print("Best Explanation:")
    print(best_explanation[0])
    print("Best Explanation Spearman: ", best_explanation[2])
    print("Best Explanation Attributions:")
    print(best_explanation[1])

    print("-" * 80)
    print("Worst Explanation:")
    print(worst_explanation[0])
    print("Worst Explanation Spearman: ", worst_explanation[2])
    print("Worst Explanation Attributions:")
    print(worst_explanation[1])

    print("=" * 80 + "\n")


def run_print_scenarios(file_path, subset=5):
    scenarios = load_jsonl(file_path)[:subset]
    # print(type(scenarios))
    # sys.exit()
    for scenario in scenarios:
        print_scenario(scenario)


if __name__ == "__main__":
    file_path = Path("dpo_datasets/codah_dpo_datasets/codah_250219_165846_LIME.jsonl")
    run_print_scenarios(str(file_path))
