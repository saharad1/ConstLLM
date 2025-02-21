from typing import Union

from torch.utils.data import Dataset

from datasets import load_dataset
from utils.data_models import ScenarioItem


class PreparedECQADataset(Dataset):
    def __init__(self, original_dataset, subset: Union[int, slice] = None):
        """
        Initialize the dataset with a specific mode.

        Args:
            original_dataset (Dataset): The original CODAH dataset.
            mode (str): The experiment mode ('exp1', 'exp2', etc.).
            subset (Union[int, slice], optional): Number or slice of examples to select.
        """
        self.original_dataset = original_dataset
        self.target_tokens: list = ["A", "B", "C", "D", "E"]

        # Subset selection
        if subset is not None:
            if isinstance(subset, int):
                self.original_dataset = self.original_dataset.select(range(subset))

        # Mode-specific instructions
        self.instruction_decision = "Choose the most plausible answer, respond only with the answer and the description:\n"
        # self.instruction_decision_2 = "Respond only with 'A', 'B', 'C', or 'D'."
        self.instruction_explain = "Why did you make that choice? Explain briefly."

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

        return self._prepare_item(example, idx)

    def _prepare_item(self, example, idx):
        question = example["q_text"]

        prompt = f"{question}\n"
        prompt += self.instruction_decision
        options = [
            example["q_op1"],
            example["q_op2"],
            example["q_op3"],
            example["q_op4"],
            example["q_op5"],
        ]
        for i, option in enumerate(options):
            prompt += f"{self.target_tokens[i]}) {option}\n"

        label = example["q_ans"]
        label_idx = options.index(label)
        label_letter = self.target_tokens[label_idx]

        # Create ScenarioItem
        user_prompts = [
            self.instruction_decision,
            self.instruction_explain,
        ]
        return ScenarioItem(
            scenario_id=idx,
            scenario_string=prompt,
            explanation_string=self.instruction_explain,
            user_prompts=user_prompts,
            label=label_letter,
        )


if __name__ == "__main__":
    ecqa_dataset = load_dataset(
        path="yangdong/ecqa", split="all", cache_dir="../datasets"
    )
    prepared_ecqa_dataset = PreparedECQADataset(ecqa_dataset, subset=10)
    print(f"Number of scenarios: {len(prepared_ecqa_dataset)}")
    print(ecqa_dataset)
    for idx, scenario_item in enumerate(prepared_ecqa_dataset, 1):
        print(f"\n=== Running Scenario {idx} ===")
        print(scenario_item)
        print(f"{scenario_item.scenario_string}")
        print(f"Correct Label: '{scenario_item.label}'")
        print("\n" + "=" * 50 + "\n")
