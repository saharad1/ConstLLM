from enum import Enum


class AttributionMethod(Enum):
    FEATURE_ABLATION = "feature_ablation"
    LIME = "lime"
    SHAPLEY_VALUE_SAMPLING = "shapley_value_sampling"
    SHAPLEY_VALUES = "shapley_values"
    KERNEL_SHAP = "kernel_shap"
    LAYER_INTEGRATED_GRADIENTS = "layer_integrated_gradients"