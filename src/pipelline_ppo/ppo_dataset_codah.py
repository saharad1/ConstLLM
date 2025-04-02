import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TextGenerationPipeline,
    pipeline,
)

from datasets import Dataset, load_dataset
from prepare_datasets.prepare_codah import PreparedCODAHDataset
from utils.custom_chat_template import custom_apply_chat_template
from utils.general import print_gpu_info


def generate_output(input_text: str, tokenizer, model) -> str:

    # Tokenize and generate
    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )

    return generated_text


def create_codah_ppo_dataset(
    model=None, tokenizer=None, subset=10, offline=False
) -> Dataset:
    if offline:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        # Ensure padding token exists
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]" if "mistral" in getattr(tokenizer, "name_or_path", "").lower() else "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            model.config.pad_token_id = tokenizer.pad_token_id

    raw_dataset = load_dataset(path="jaredfern/codah", name="codah", split="all")
    prepared_dataset = PreparedCODAHDataset(raw_dataset, subset=subset)
    ppo_dataset = []

    for iteration, scenario_item in tqdm(
        enumerate(prepared_dataset, 1),
        total=len(prepared_dataset),
        desc="Preparing PPO Dataset",
    ):
        # Decision Phase
        decision_prompt = custom_apply_chat_template(
            [{"role": "user", "content": scenario_item.scenario_string}]
        )

        decision_output = generate_output(decision_prompt, tokenizer, model)

        # Explanation Phase
        explanation_conv = [
            {"role": "user", "content": scenario_item.scenario_string},
            {"role": "assistant", "content": decision_output},
            {"role": "user", "content": scenario_item.explanation_string},
        ]

        # Append structured data to dataset
        ppo_dataset.append({"query": explanation_conv})
        # explanation_prompt = custom_apply_chat_template(explanation_conv)

    return ppo_dataset


def save_ppo_dataset_jsonl(dataset, file_path="ppo_dataset.jsonl"):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")  # Each JSON object is written on a new line
    print(f"Dataset saved to {file_path} in JSONL format")


def load_ppo_dataset_jsonl(file_path="ppo_dataset.jsonl"):
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    print(f"Dataset loaded from {file_path} in JSONL format")
    return dataset


if __name__ == "__main__":
    ppo_dataset = create_codah_ppo_dataset(subset=None, offline=True)
    ppo_dataset_path = Path("ppo_datasets") / "ppo_codah_dataset.jsonl"
    save_ppo_dataset_jsonl(ppo_dataset, file_path=str(ppo_dataset_path))
