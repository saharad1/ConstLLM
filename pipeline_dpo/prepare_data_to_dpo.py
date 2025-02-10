import json


def convert_to_dpo_format(input_path, output_path, include_scores=False):
    """
    Converts the input JSONL dataset to the Direct Preference Optimization (DPO) format.
    Optionally includes scores if `include_scores` is set to True.
    """
    dpo_data = []

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            item = json.loads(line.strip())

            # Construct the conversation-based prompt
            prompt = [
                {"role": "user", "content": item["decision_prompt"]},
                {"role": "assistant", "content": item["decision_output"]},
                {"role": "user", "content": item["explanation_prompt"]},
            ]

            # Prepare the chosen and rejected responses (assistant outputs)
            dpo_entry = {
                "prompt": prompt,
                "chosen": item["explanation_best_output"],
                "rejected": item["explanation_worst_output"],
            }

            # Optionally add scores
            if include_scores:
                dpo_entry["score_chosen"] = item["explanation_best_score"]
                dpo_entry["score_rejected"] = item["explanation_worst_score"]

            dpo_data.append(dpo_entry)

    # Write the new dataset in JSONL format
    with open(output_path, "w", encoding="utf-8") as outfile:
        for entry in dpo_data:
            json.dump(entry, outfile)
            outfile.write("\n")

    print(f"✅ DPO dataset saved to {output_path} (Scores included: {include_scores})")


# Run the conversion
if "__name__" == "__main__":
    # Input and output file paths
    input_file = "input.jsonl"  # Replace with your actual file path
    output_file = "dpo_dataset.jsonl"
    convert_to_dpo_format(input_file, output_file)
