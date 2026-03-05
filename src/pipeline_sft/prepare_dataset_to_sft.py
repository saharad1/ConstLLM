import json
from pathlib import Path
from typing import Optional

from datasets import Dataset


def load_sft_dataset(
    file_path,
    diff_threshold: Optional[float] = None,
    similarity_metric: str = "spearman",
) -> Dataset:
    """
    Load a JSONL dataset into a Hugging Face Dataset with "messages" format for SFTTrainer.
    Uses the same JSONL format as the DPO pipeline; each example is the full conversation
    with the chosen (best) explanation only.

    Args:
        file_path: Path to the JSONL file.
        diff_threshold: Optional minimum threshold for difference between best and worst
                        scores. If provided, filters out examples below this threshold.
        similarity_metric: Which similarity metric to use for "best" explanation.
                           Options: "spearman" or "cosine".

    Returns:
        A Hugging Face Dataset with a "messages" column (list of {role, content} dicts).
    """
    if similarity_metric not in ["spearman", "cosine"]:
        raise ValueError(f"Invalid similarity metric: {similarity_metric}. Must be 'spearman' or 'cosine'.")

    best_key = f"{similarity_metric}_best"
    diff_key = f"{similarity_metric}_score_diff"

    data_list = []
    total_count = 0
    kept_count = 0
    filtered_count = 0

    print(f"Loading SFT dataset from {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    total_count += 1

                    if diff_threshold is not None:
                        score_diff = item.get(diff_key, 0)
                        if score_diff < diff_threshold:
                            filtered_count += 1
                            continue

                    messages = [
                        {"role": "user", "content": item["decision_prompt"]},
                        {"role": "assistant", "content": item["decision_output"]},
                        {"role": "user", "content": item["explanation_prompt"]},
                        {"role": "assistant", "content": item[best_key]["explanation_output"]},
                    ]
                    data_list.append({"messages": messages})
                    kept_count += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_idx+1}: {str(e)}")
                except Exception as e:
                    print(f"Error processing line {line_idx+1}: {str(e)}")
    except Exception as e:
        print(f"Error opening file {file_path}: {str(e)}")

    dataset = Dataset.from_list(data_list)

    if diff_threshold is not None:
        print(f"Total examples: {total_count}")
        print(f"Filtered using {similarity_metric} similarity:")
        print(f"\t- Filtered out {filtered_count} examples with score difference below {diff_threshold}")
        print(f"\t- Kept {kept_count} examples with score difference >= {diff_threshold}")
    else:
        print(f"Loaded {kept_count} examples from {file_path}")

    return dataset
