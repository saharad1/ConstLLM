import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from src.utils.data_models import Choice75ScenarioItem

DATASET_PATH = Path("datasets") / "choice-75"


def load_json_files(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith(".json"):
            with open(os.path.join(directory, file), "r") as f:
                data.append(json.load(f))
    return data


class PreparedCHOICE75Dataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, subset: int = None):
        self.dataset_path = dataset_path
        self.original_choice75_data = []

        self.target_tokens: list = ["A", "B", "C"]
        self.labels_to_tokens = {1: "A", 2: "B", 0: "C"}

        self.choice75_scenarios = []
        self.exp1_instruction = "Given the Scenario, which option is the better choice in order to achieve the Goal? respond only with the answer and the description:\n"
        self.exp1_explain_prompt = "Why did you make that choice? Explain briefly."

        self.load_dataset()
        self.divide_to_scenarios()

        if subset is not None:
            if isinstance(subset, int):
                self.choice75_scenarios = self.choice75_scenarios[:subset]

    def load_dataset(self):
        # Load verb_phrase_manual (if needed)
        verb_phrase_manual_train = load_json_files(self.dataset_path / "verb_phrase_manual" / "train")
        verb_phrase_manual_dev = load_json_files(self.dataset_path / "verb_phrase_manual" / "dev")
        verb_phrase_manual = verb_phrase_manual_train + verb_phrase_manual_dev

        # Load verb_phrase_machine (if needed)
        verb_phrase_machine_train = load_json_files(self.dataset_path / "verb_phrase_machine" / "train")
        verb_phrase_machine_dev = load_json_files(self.dataset_path / "verb_phrase_machine" / "dev")
        verb_phrase_machine = verb_phrase_machine_train + verb_phrase_machine_dev

        # Load train and dev splits from user_profile
        user_profile_train = load_json_files(self.dataset_path / "user_profile" / "train")
        user_profile_dev = load_json_files(self.dataset_path / "user_profile" / "dev")
        user_profile = user_profile_train + user_profile_dev

        self.original_choice75_data = verb_phrase_manual + verb_phrase_machine + user_profile
        self.original_choice75_data.sort(key=lambda x: x["index"])

    def divide_to_scenarios(self):
        index = 0
        for entry in self.original_choice75_data:
            # Extract the goal
            goal = entry["goal"]
            branching_idx = entry["branching_info"]["branching_idx"]
            options = [
                entry["branching_info"]["option 1"],
                entry["branching_info"]["option 2"],
                "Either one, since they have similar effect when it comes to the goal",
            ]
            options_string = "\n".join(f"{self.target_tokens[i]}) {option}" for i, option in enumerate(options))

            steps = entry["steps"][:branching_idx]

            # Combine the goal and steps into a single string
            scenario_base_string = f"Goal: {goal}. \nSteps Taken: {' -> '.join(steps)}\nNext Step: {entry['steps'][branching_idx]}"
            for scenario in entry["branching_info"]["freeform_ra"]:
                scenario_text = scenario[0]
                scenario_label = self.labels_to_tokens[scenario[1]]
                scenario_difficulty = scenario[2]
                scenario_string = f"{scenario_base_string}\nScenario: {scenario_text}\n{self.exp1_instruction}{options_string}"
                self.choice75_scenarios.append(
                    Choice75ScenarioItem(
                        scenario_id=index,
                        scenario_string=scenario_string,
                        user_prompts=[self.exp1_instruction, self.exp1_explain_prompt],
                        explanation_string=self.exp1_instruction,
                        label=scenario_label,
                        difficulty=scenario_difficulty,
                    )
                )
                index += 1

    def get_exp1_static_texts(self):
        static_texts_decision = [self.exp1_instruction]
        static_texts_explanation = [self.exp1_instruction, self.exp1_explain_prompt]
        return static_texts_decision, static_texts_explanation

    def __len__(self):
        return len(self.choice75_scenarios)

    def __getitem__(self, idx):
        return self.choice75_scenarios[idx]


if __name__ == "__main__":
    choice75_dataset = PreparedCHOICE75Dataset()
    print(choice75_dataset.choice75_scenarios[0].scenario_string)
    print(len(choice75_dataset))
    print(choice75_dataset.choice75_scenarios[4].scenario_string)
