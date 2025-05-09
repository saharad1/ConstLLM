from typing import Union

from datasets import load_dataset
from torch.utils.data import Dataset

from src.utils.data_models import ScenarioItem


class PreparedARCDataset(Dataset):
    def __init__(self, original_dataset, subset: Union[int, slice] = None):
        """
        Initialize the dataset with a specific mode.

        Args:
            original_dataset (Dataset): The original CODAH dataset.
            subset (Union[int, slice], optional): Number or slice of examples to select.
        """
        self.original_dataset = original_dataset
        self.target_tokens: list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        # Subset selection
        if subset is not None:
            if isinstance(subset, int):
                self.original_dataset = self.original_dataset.select(range(subset))

        # Mode-specific instructions
        self.instruction_decision = "Choose the most plausible answer. Respond only with the letter and the full answer text from the list below:\n"
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
        question = example["question"]

        prompt = f"{question}\n"
        prompt += self.instruction_decision

        options = example["choices"]["text"]
        for i, option in enumerate(options):
            prompt += f"{self.target_tokens[i]}) {option}\n"

        label_letter = example["answerKey"]
        # label_idx = options.index(label)
        # label_letter = self.target_tokens[label_idx]

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
    arc_dataset = load_dataset(path="allenai/ai2_arc", name="ARC-Challenge", split="all")

    prepared_arc_dataset = PreparedARCDataset(arc_dataset, subset=None)
    print(f"Number of scenarios: {len(prepared_arc_dataset)}")
    print(arc_dataset)
    for idx, scenario_item in enumerate(prepared_arc_dataset, 1):
        print(f"\n=== Running Scenario {idx} ===")
        print(scenario_item)
        print(f"{scenario_item.scenario_string}")
        print(f"Correct Label: '{scenario_item.label}'")
        print("\n" + "=" * 50 + "\n")

    print(f"Number of scenarios: {len(arc_dataset)}")
    print(f"Number of scenarios: {len(prepared_arc_dataset)}")
