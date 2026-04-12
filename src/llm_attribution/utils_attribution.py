from enum import Enum


class AttributionMethod(Enum):
    FEATURE_ABLATION = "feature_ablation"
    LIME = "lime"
    SHAPLEY_VALUE_SAMPLING = "shapley_value_sampling"
    SHAPLEY_VALUES = "shapley_values"
    KSHAP = "kshap"
    LIG = "layer_integrated_gradients"
