from llm_attribution.utils_attribution import AttributionMethod
from utils.custom_chat_template import custom_apply_chat_template
from utils.phase_run import MethodParams


def reward_fn():
    attribution_method = AttributionMethod.LIME.name

    # Set parameters using a single function call per method
    if attribution_method == AttributionMethod.LIME.name:
        methods_params_decision = MethodParams.set_params(
            AttributionMethod.LIME.name, n_samples=500, perturbations_per_eval=500
        )
        methods_params_explanation = MethodParams.set_params(
            AttributionMethod.LIME.name, n_samples=500, perturbations_per_eval=500
        )
    elif attribution_method == AttributionMethod.LIG.name:
        methods_params_decision = MethodParams.set_params(
            AttributionMethod.LIG.name, n_steps=30
        )
        methods_params_explanation = MethodParams.set_params(
            AttributionMethod.LIG.name, n_steps=30
        )
        device = "auto"
    else:
        raise ValueError(f"Invalid attribution method: {attribution_method}")

    # Decision Phase
    decision_prompt = custom_apply_chat_template(
        [{"role": "user", "content": scenario_item.scenario_string}]
    )
    decision_output, decision_result = run_phase(
        llm_analyzer=llm_analyzer,
        prompt=decision_prompt,
        methods_params=methods_params_decision,
        phase="decision",
    )

    decision_attributions = decision_result.methods_scores[current_method]

    # Explanation Phase
    explanation_prompt = custom_apply_chat_template(
        [
            {"role": "user", "content": scenario_item.scenario_string},
            {"role": "assistant", "content": decision_output},
            {"role": "user", "content": scenario_item.explanation_string},
        ]
    )
    explanation_output, explanation_result = run_phase(
        llm_analyzer=llm_analyzer,
        prompt=explanation_prompt,
        methods_params=methods_params_explanation,
        phase="explanation",
    )

    explanation_attributions = explanation_result.methods_scores[current_method]

    # Compute Spearman score
    curr_spearman_score = compute_spearman_score(
        decision_attributions=decision_attributions,
        explanation_attributions=explanation_attributions,
    )
