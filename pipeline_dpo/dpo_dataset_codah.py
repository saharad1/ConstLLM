import json

import torch
from torch.utils.data import DataLoader, Dataset

from datasets import Dataset, load_dataset


def load_dpo_dataset(file_path, include_scores=False):
    """
    Efficiently loads a JSONL dataset into a Hugging Face Dataset format for DPOTrainer.
    Uses `load_dataset` for direct streaming and minimal memory usage.
    """

    def process_example(item):
        """
        Function to convert a single JSON example into the DPO format.
        """
        prompt = [
            {"role": "user", "content": item["decision_prompt"]},
            {"role": "assistant", "content": item["decision_output"]},
            {"role": "user", "content": item["explanation_prompt"]},
        ]

        dpo_entry = {
            "prompt": prompt,
            "chosen": [
                {"role": "assistant", "content": item["explanation_best_output"]}
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": item["explanation_worst_output"],
                }
            ],
        }

        if include_scores:
            dpo_entry["score_chosen"] = item["explanation_best_score"]
            dpo_entry["score_rejected"] = item["explanation_worst_score"]

        return dpo_entry

    # Load dataset efficiently (streaming mode)
    dataset = load_dataset("json", data_files=file_path, split="train")

    # Apply processing function lazily
    return dataset.map(process_example, remove_columns=dataset.column_names)


# class DPODatasetCodah(Dataset):
#     """
#     PyTorch dataset for Direct Preference Optimization (DPO).
#     Loads a dataset in the required format for Hugging Face `trl`.
#     """

#     def __init__(self, data_path, include_scores=False):
#         self.data = self.load_dpo_data(data_path, include_scores)

#     def load_dpo_data(self, file_path, include_scores):
#         """Load and process JSONL data into DPO format."""
#         dpo_data = []
#         with open(file_path, "r", encoding="utf-8") as infile:
#             for line in infile:
#                 item = json.loads(line.strip())

#                 # Create conversation-based prompt
#                 prompt = [
#                     {"role": "user", "content": item["decision_prompt"]},
#                     {"role": "assistant", "content": item["decision_output"]},
#                     {"role": "user", "content": item["explanation_prompt"]},
#                 ]

#                 # Create dataset entry
#                 dpo_entry = {
#                     "prompt": prompt,
#                     "chosen": item["explanation_best_output"],
#                     "rejected": item["explanation_worst_output"],
#                 }

#                 # Optionally add scores
#                 if include_scores:
#                     dpo_entry["score_chosen"] = item["explanation_best_score"]
#                     dpo_entry["score_rejected"] = item["explanation_worst_score"]

#                 dpo_data.append(dpo_entry)

#         return dpo_data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


# # Load dataset as a PyTorch DataLoader
# def get_dpo_dataloader(file_path, batch_size=4, include_scores=False):
#     dataset = DPODatasetCodah(file_path, include_scores)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Example usage
if __name__ == "__main__":
    # # Input file path (change to your dataset location)
    # input_file = "results/codah_res/codah_results2.jsonl"
    # dpo_dataloader = get_dpo_dataloader(input_file, batch_size=4, include_scores=False)

    # # Iterate through one batch
    # for batch in dpo_dataloader:
    #     print(batch)
    #     break  # Print one batch and stop

    train_dataset = load_dpo_dataset(
        "results/codah_res/codah_results2.jsonl", include_scores=True
    )

    # Check a sample
    print(train_dataset[0])
