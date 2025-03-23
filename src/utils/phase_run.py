from src.llm_attribution.utils_attribution import AttributionMethod
from src.utils.data_models import LLMAnalysisRes


class MethodParams:
    _METHOD_PARAMS_FUNCTIONS = {
        AttributionMethod.LIME.name: lambda n_samples=500, perturbations_per_eval=250: {
            AttributionMethod.LIME.name: {
                "n_samples": n_samples,
                "perturbations_per_eval": perturbations_per_eval,
            }
        },
        AttributionMethod.LIG.name: lambda n_steps=50: {
            AttributionMethod.LIG.name: {
                "n_steps": n_steps,
            }
        },
        AttributionMethod.SHAPLEY_VALUE_SAMPLING.name: lambda n_samples=50, perturbations_per_eval=50: {
            AttributionMethod.SHAPLEY_VALUE_SAMPLING.name: {
                "n_samples": n_samples,
                "perturbations_per_eval": perturbations_per_eval,
            }
        },
        AttributionMethod.FEATURE_ABLATION.name: lambda perturbations_per_eval=50: {
            AttributionMethod.FEATURE_ABLATION.name: {
                "perturbations_per_eval": perturbations_per_eval,
            }
        },
    }

    @classmethod
    def set_params(cls, method_name: str, **kwargs):
        """Sets the parameters for a specific attribution method."""
        if method_name not in cls._METHOD_PARAMS_FUNCTIONS:
            raise ValueError(f"Invalid method name: {method_name}")
        return cls._METHOD_PARAMS_FUNCTIONS[method_name](**kwargs)


# Generalized phase function
def run_phase(llm_analyzer, prompt, methods_params, phase="decision", pre_generated_output=None, pre_generated_attributions=None):
    print(f"Running {phase} phase...")

    # Use pre-generated output if provided, otherwise generate it
    if pre_generated_output is not None:
        output = pre_generated_output
        print(f"Using pre-generated output for {phase} phase")
    else:
        output = llm_analyzer.generate_output(prompt)

    if pre_generated_attributions is not None:
        print(f"Using pre-generated attributions for {phase} phase")

        # Check if pre_generated_attributions is already an LLMAnalysisRes object
        if isinstance(pre_generated_attributions, LLMAnalysisRes):
            result = pre_generated_attributions
        else:
            # If it's just the attribution scores, create an LLMAnalysisRes object
            # Assuming the first key in methods_params is the method name
            method_name = next(iter(methods_params))

            # Create a methods_scores dictionary with the method name as key
            # Assuming pre_generated_attributions is a list of attribution scores
            methods_scores = {method_name: pre_generated_attributions}

            # Create the LLMAnalysisRes object
            result = LLMAnalysisRes(input_text=prompt, target=output, methods_scores=methods_scores)
    else:
        result = llm_analyzer.analyze(input_text=prompt, target=output, method_params=methods_params)

    if not result:
        raise ValueError(f"{phase.capitalize()} phase returned invalid results")
    return output, result
