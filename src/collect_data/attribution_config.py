# src/collect_data/attribution_config.py
from src.llm_attribution.utils_attribution import AttributionMethod
from src.utils.phase_run import MethodParams


def configure_attribution_methods(attribution_method_name, phase=None):
    """
    Configure attribution method parameters for different phases.

    Args:
        attribution_method_name: Name of the attribution method to configure
        phase: Optional phase name to configure differently (decision/explanation)

    Returns:
        Method parameters dictionary
    """
    if attribution_method_name == AttributionMethod.LIME.name:
        params = MethodParams.set_params(AttributionMethod.LIME.name, n_samples=500, perturbations_per_eval=500)
    elif attribution_method_name == AttributionMethod.LIG.name:
        params = MethodParams.set_params(AttributionMethod.LIG.name, n_steps=25)
    elif attribution_method_name == AttributionMethod.SHAPLEY_VALUE_SAMPLING.name:
        params = MethodParams.set_params(AttributionMethod.SHAPLEY_VALUE_SAMPLING.name, n_samples=500)
    else:
        raise ValueError(f"Unsupported attribution method: {attribution_method_name}")

    return {attribution_method_name: params}
