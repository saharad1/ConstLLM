"""
Configuration for attribution methods used in data collection.
"""

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
    # Normalize the attribution method name to match the enum values
    attribution_method_upper = attribution_method_name.upper()

    if attribution_method_upper == "LIME":
        # Use the actual enum value for consistency
        method_name = AttributionMethod.LIME.name
        params = MethodParams.set_params(method_name, n_samples=500, perturbations_per_eval=500)
    elif attribution_method_upper == "LIG":
        method_name = AttributionMethod.LIG.name
        params = MethodParams.set_params(method_name, n_steps=25)
    elif attribution_method_upper == "SHAPLEY_VALUE_SAMPLING":
        method_name = AttributionMethod.SHAPLEY_VALUE_SAMPLING.name
        params = MethodParams.set_params(method_name, n_samples=500)
    else:
        raise ValueError(f"Unsupported attribution method: {attribution_method_name}")

    # Return the params directly - they already have the method name as a key
    return params


def get_attribution_methods_params(attribution_method_name):
    """
    Get attribution method parameters for both decision and explanation phases.

    Args:
        attribution_method_name: Name of the attribution method to configure

    Returns:
        Tuple of (decision_params, explanation_params)
    """
    # Configure parameters for decision phase
    methods_params_decision = configure_attribution_methods(attribution_method_name, "decision")

    # Configure parameters for explanation phase
    methods_params_explanation = configure_attribution_methods(attribution_method_name, "explanation")

    return methods_params_decision, methods_params_explanation
