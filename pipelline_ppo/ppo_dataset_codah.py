import os

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TextGenerationPipeline,
    pipeline,
)

from datasets import load_dataset
from prepare_datasets.prepare_codah import PreparedCODAHDataset
from utils.custom_chat_template import custom_apply_chat_template
from utils.general import print_gpu_info

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print_gpu_info()


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


def create_codah_ppo_dataset():
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
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id

    subset = 10
    raw_dataset = load_dataset(path="jaredfern/codah", name="codah", split="all")
    prepared_dataset = PreparedCODAHDataset(raw_dataset, subset=subset)

    for iteration, scenario_item in tqdm(
        enumerate(prepared_dataset, 1),
        total=len(prepared_dataset),
        desc="Processing Scenarios",
    ):
        # Decision Phase
        decision_prompt = custom_apply_chat_template(
            [{"role": "user", "content": scenario_item.scenario_string}]
        )

        decision_output = generate_output(decision_prompt, tokenizer, model)

        # Explanation Phase
        explanation_prompt = custom_apply_chat_template(
            [
                {"role": "user", "content": scenario_item.scenario_string},
                {"role": "assistant", "content": decision_output},
                {"role": "user", "content": scenario_item.explanation_string},
            ]
        )
        print(explanation_prompt)
        break


if __name__ == "__main__":
    create_codah_ppo_dataset()
