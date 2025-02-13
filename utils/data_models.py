from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from captum.attr._core.llm_attr import LLMAttributionResult


@dataclass
class ScenarioItem:
    scenario_id: int
    scenario_string: str
    user_prompts: List[str]
    explanation_string: str
    label: str


@dataclass
class ScenarioSummary:
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
    explanation_scores: Dict[
        str, float
    ]  # Scores for each method in the explanation phase
    extra_info: Dict[str, Any] = field(
        default_factory=dict
    )  # Flexible field for extra dataset-specific information

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
class ExplanationRanking:
    decision_output: str
    explanation_output: str
    spearman_score: float


@dataclass
class ScenarioScores:
    """
    Represents the result of a scenario, including decision and explanation phases.
    Easily extendable with additional fields for specific datasets.
    """

    scenario_id: int
    correct_label: str
    decision_prompt: str
    decisions_outputs: List[str]
    explanation_prompt: str
    decision_attributions: List[Dict[str, float]]
    explanation_attributions: List[Dict[str, float]]
    explanation_outputs: List[str]
    spearman_scores: List[float]
    explanation_best: ExplanationRanking
    explanation_worst: ExplanationRanking


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
    explanation_prompt: str  # Details of the explanation phase
    explanation_best_output: str  # Prompt input for the explanation phase
    explanation_best_score: float  # Score for the best method in the explanation phase
    explanation_worst_output: str  # Prompt input for the explanation phase
    explanation_worst_score: str  # Prompt input for the explanation phase

    def to_dict(self):
        return {
            "scenario_id": self.scenario_id,
            "correct_label": self.correct_label,
            "decision_prompt": self.decision_prompt,
            "decision_output": self.decision_output,
            "explanation_prompt": self.explanation_prompt,
            "explanation_best_output": self.explanation_best_output,
            "explanation_best_score": self.explanation_best_score,
            "explanation_worst_output": self.explanation_worst_output,
            "explanation_worst_score": self.explanation_worst_score,
        }


@dataclass
class LLMAnalysisRes:
    """
    Represents the result of a scenario, including decision and explanation phases.
    Easily extendable with additional fields for specific datasets.
    """

    input_text: str
    target: str
    methods_scores: Dict[str, Any]


@dataclass
class Choice75ScenarioItem:
    scenario_id: int
    scenario_string: str
    user_prompts: List[str]
    explanation_string: str
    label: str
    difficulty: str
