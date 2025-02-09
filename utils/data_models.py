from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from captum.attr._core.llm_attr import LLMAttributionResult


@dataclass
class ScenarioItem:
    scenario_id: int
    scenario_string: str
    user_prompts: List[str]
    label: str

@dataclass
class ScenarioResult:
    """
    Represents the result of a scenario, including decision and explanation phases.
    Easily extendable with additional fields for specific datasets.
    """
    scenario_id: int  # Unique scenario ID
    correct_label: str  # Ground truth label
    decision_prompt: str  # Prompt input for the decision phase
    decision_output: str  # Details of the decision phase
    decision_scores: Dict[str, float]  # Scores for each method in the decision phase
    explanation_prompt: str  # Details of the explanation phase
    explanation_output: str  # Prompt input for the explanation phase
    explanation_scores: Dict[str, float]  # Scores for each method in the explanation phase
    extra_info: Dict[str, Any] = field(default_factory=dict)  # Flexible field for extra dataset-specific information

    # def to_dict(self) -> Dict[str, Any]:
    #     """
    #     Converts the ScenarioResult instance into a comprehensive dictionary format.
    #     """
    #     result_dict = {
    #         "scenario_id": self.scenario_id,
    #         "correct_label": self.correct_label,
    #         "decision": {
    #             "prompt_input": self.decision.prompt_input,
    #             "model_output": self.decision.model_output,
    #             "methods_scores": self.decision.methods_scores,
    #         },
    #         "explanation": {
    #             "prompt_input": self.explanation.prompt_input,
    #             "model_output": self.explanation.model_output,
    #             "methods_scores": self.explanation.methods_scores,
    #         },
    #     }
    #     if self.extra_info:
    #         result_dict.update(self.extra_info)
    #     return result_dict

    def print_results(self):
        """
        Pretty-prints the scenario results, including flexible extra fields.
        """
        print(f"=== SCENARIO {self.scenario_id} RESULTS ===")
        print(f"Decision Prompt: {self.decision_prompt}")
        print(f"Decision Output: {self.decision_output}")
        print(f"Correct Label: {self.correct_label}")
        print(f"Explanation Prompt: {self.explanation_prompt}")
        print(f"Explanation Output: {self.explanation_output}")
        print("Decision Scores:")
        for method, scores in self.decision_scores.items():
            print(f"  {method}: {scores}")
        print("Explanation Scores:")
        for method, scores in self.explanation_scores.items():
            print(f"  {method}: {scores}")
        if self.extra_info:
            print("\n--- EXTRA INFORMATION ---")
            for key, value in self.extra_info.items():
                print(f"{key}: {value}")
                
@dataclass
class LLMAnalysisRes:
    """
    Represents the result of a scenario, including decision and explanation phases.
    Easily extendable with additional fields for specific datasets.
    """
    input_text : str
    target : str
    methods_scores : Dict[str,Any]
