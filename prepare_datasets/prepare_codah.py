from collections import namedtuple
from typing import Union

from datasets import Dataset, load_dataset
from torch.utils.data import Dataset

# ScenarioItem = namedtuple('ScenarioItem', ['user_prompts', 'label'])
from utils.data_models import ScenarioItem


class PreparedCODAHDataset(Dataset):
    def __init__(
        self, original_dataset, mode: str = "exp1", subset: Union[int, slice] = None
    ):
        """
        Initialize the dataset with a specific mode.

        Args:
            original_dataset (Dataset): The original CODAH dataset.
            mode (str): The experiment mode ('exp1', 'exp2', etc.).
            subset (Union[int, slice], optional): Number or slice of examples to select.
        """
        self.original_dataset = original_dataset
        self.mode = mode.lower()
        self.target_tokens: list = ["A", "B", "C", "D"]

        # Subset selection
        if subset is not None:
            if isinstance(subset, int):
                self.original_dataset = self.original_dataset.select(range(subset))

        # Mode-specific instructions
        if self.mode == "exp1":
            self.instruction_decision = "Choose the most plausible answer, respond only with the answer and the description:\n"
            self.instruction_decision_2 = "Respond only with 'A', 'B', 'C', or 'D'."
            self.instruction_explain = "Why did you make that choice? Explain briefly."
        elif self.mode == "exp2":
            self.instruction_decision = "Choose the most plausible answer, provide your response in the following format:\n"
            self.instruction_decision_2 = "Decision: <The letter of your chosen answer followed by the full text of the chosen option>\nExplanation: <Your explanation>\n"
        else:
            raise ValueError(f"Mode '{self.mode}' is not supported.")

    def get_exp1_static_texts(self):
        """Return static texts for decision and explanation phases."""
        static_texts_decision = [self.instruction_decision, self.instruction_decision_2]
        static_texts_explanation = [
            self.instruction_decision,
            self.instruction_decision_2,
            self.instruction_explain,
        ]
        return static_texts_decision, static_texts_explanation

    def get_exp2_static_texts(self):
        return [self.instruction_decision, self.instruction_decision_2]

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        """
        Generate a prompt and scenario item based on the mode.

        Args:
            idx (int): Index of the dataset item.

        Returns:
            ScenarioItem: A structured item containing the prompt, user prompts, and label.
        """
        example = self.original_dataset[idx]

        if self.mode == "exp1":
            return self._prepare_exp1_item(example, idx)
        elif self.mode == "exp2":
            return self._prepare_exp2_item(example, idx)
        else:
            raise ValueError(f"Mode '{self.mode}' is not supported.")

    def _prepare_exp1_item(self, example, idx):
        """Prepare an item for Experiment 1 (Exp1)."""
        question = example["question_propmt"]
        options = example["candidate_answers"]

        # Generate the decision prompt
        prompt = f"{question}\n"
        prompt += self.instruction_decision
        prompt += "\n".join(
            f"{self.target_tokens[i]}) {option}" for i, option in enumerate(options)
        )
        prompt += "\n" + self.instruction_decision_2

        # Get the correct answer
        label = example["correct_answer_idx"]
        label_letter = self.target_tokens[label]

        # Create ScenarioItem
        user_prompts = [
            self.instruction_decision + self.instruction_decision_2,
            self.instruction_explain,
        ]
        return ScenarioItem(
            scenario_id=idx,
            scenario_string=prompt,
            explanation_string=self.instruction_explain,
            user_prompts=user_prompts,
            label=label_letter,
        )

    def _prepare_exp2_item(self, example, idx):
        """Prepare an item for Experiment 2 (Exp2)."""
        question = example["question_propmt"]
        options = example["candidate_answers"]

        # Generate the decision prompt
        prompt = f"{self.instruction_decision}"
        prompt += f"{self.instruction_decision_2}"
        prompt += f"{question}\n"
        prompt += "\n".join(
            f"{self.target_tokens[i]}) {option}" for i, option in enumerate(options)
        )

        # Get the correct answer
        label = example["correct_answer_idx"]
        label_letter = self.target_tokens[label]

        # Create ScenarioItem
        user_prompts = [self.instruction_decision]
        return ScenarioItem(
            scenario_id=idx,
            scenario_string=prompt,
            explanation_string=self.instruction_explain,
            user_prompts=user_prompts,
            label=label_letter,
        )


def show_codah_data():
    # Load the CODAH dataset
    codah_dataset: Dataset = load_dataset(
        path="jaredfern/codah", name="codah", split="all", cache_dir="../datasets"
    )

    print(f"Number of scenarios: {len(codah_dataset)}")

    # Prepare scenarios from the dataset
    prepared_codah_dataset = PreparedCODAHDataset(codah_dataset, mode="exp2", subset=5)
    print(f"Number of scenarios: {len(prepared_codah_dataset)}")

    for idx, scenario_item in enumerate(prepared_codah_dataset, 1):
        print(f"\n=== Running Scenario {idx} ===")
        print(scenario_item)
        print(f"{scenario_item.scenario_string}")
        print(f"Correct Label: '{scenario_item.label}'")
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    print("Running example usage...")
    show_codah_data()
