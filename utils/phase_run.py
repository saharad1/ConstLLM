from llm_attribution.utils_attribution import AttributionMethod


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
    }

    @classmethod
    def set_params(cls, method_name: str, **kwargs):
        """Sets the parameters for a specific attribution method."""
        if method_name not in cls._METHOD_PARAMS_FUNCTIONS:
            raise ValueError(f"Invalid method name: {method_name}")
        return cls._METHOD_PARAMS_FUNCTIONS[method_name](**kwargs)


# Generalized phase function
def run_phase(llm_analyzer, prompt, methods_params, phase="decision"):
    print(f"Running {phase} phase...")
    output = llm_analyzer.generate_output(prompt)
    result = llm_analyzer.analyze(
        input_text=prompt, target=output, method_params=methods_params
    )
    if not result:
        raise ValueError(f"{phase.capitalize()} phase returned invalid results")
    return output, result
