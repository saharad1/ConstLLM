import sys
import time

from datasets import load_dataset
from tqdm import tqdm

from llm_attribution.LLMAnalyzer import LLMAnalyzer
from llm_attribution.utils_attribution import AttributionMethod
from pipeline_dpo.comp_score import compute_kl_divergence, compute_spearman_score
from prepare_datasets.prepare_codah import PreparedCODAHDataset
from utils.cutom_chat_template import custom_apply_chat_template
from utils.data_models import ScenarioResult

raw_dataset = load_dataset(path="jaredfern/codah", name="codah", split="all")
prepared_dataset = PreparedCODAHDataset(raw_dataset, mode="exp1", subset=10)
print(f"Number of scenarios: {len(prepared_dataset)}")

llm_analyzer = LLMAnalyzer(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")

methods_params_decision = {
    AttributionMethod.LIME.name: {
        "n_samples": 20,
        "perturbations_per_eval": 20,
    }
}
methods_params_explanation = {
    AttributionMethod.LIME.name: {
        "n_samples": 20,
        "perturbations_per_eval": 20,
    }
}

for idx, scenario_item in tqdm(
    enumerate(prepared_dataset, 1),
    total=len(prepared_dataset),
    desc="Processing Scenarios",
):
    iteration_start_time = time.time()
    print(f"\n=== Running Scenario {idx} ===")
    # Decision Phase
    print("=== DECISION PHASE ===")
    decision_prompt = custom_apply_chat_template(
        [{"role": "user", "content": scenario_item.scenario_string}]
    )
    # print(decision_prompt)
    decision_output = llm_analyzer.generate_output(decision_prompt)
    # print(decision_output)

    decision_result = llm_analyzer.analyze(
        input_text=decision_prompt,
        target=decision_output,
        method_params=methods_params_decision,
        static_texts=None,
    )

    # print(f"decision results: {decision_result}")

    if not decision_result:
        raise ValueError(
            f"Decision phase returned invalid or empty methods_scores for scenario {idx}"
        )

    # Explanation Phase
    print("=== EXPLANATION PHASE ===")
    explanation_prompt = custom_apply_chat_template(
        [
            {"role": "user", "content": scenario_item.scenario_string},
            {"role": "assistant", "content": decision_output},
            {"role": "user", "content": scenario_item.user_prompts[1]},
        ]
    )

    explanation_output = llm_analyzer.generate_output(explanation_prompt)
    explanation_result = llm_analyzer.analyze(
        input_text=explanation_prompt,
        target=explanation_output,
        method_params=methods_params_explanation,
        static_texts=None,
    )

    # print(f"exp results: {explanation_result}")
    # Validate explanation results
    if not explanation_result:
        raise ValueError(
            f"Explanation phase returned invalid or empty methods_scores for scenario {idx}"
        )

    # Create a ScenarioResult object
    scenario_result = ScenarioResult(
        scenario_id=idx,
        correct_label=scenario_item.label,
        decision_prompt=scenario_item.scenario_string,
        decision_output=decision_output,
        decision_scores=decision_result.methods_scores,
        explanation_prompt=scenario_item.user_prompts[1],
        explanation_output=explanation_output,
        explanation_scores=explanation_result.methods_scores,
    )
    scenario_result.print_results()

    decisions_attributions = scenario_result.decision_scores[
        AttributionMethod.LIME.name
    ]
    explanations_attributions = scenario_result.explanation_scores[
        AttributionMethod.LIME.name
    ]

    spearman_score = compute_spearman_score(
        decision_attributions=decisions_attributions,
        explanation_attributions=explanations_attributions,
    )

    # kl_divergance_score = compute_kl_divergence(
    #     decision_attributions=decisions_attributions,
    #     explanation_attributions=explanations_attributions,
    # )

    print(f"\nSpearman Score: {spearman_score}")
    # print(f"KL Divergance Score: {kl_divergance_score}")
