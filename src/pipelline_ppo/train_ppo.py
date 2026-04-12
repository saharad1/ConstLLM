import os
import pickle
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
    TextGenerationPipeline,
    pipeline,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from datasets import Dataset
from llm_attribution.LLMAnalyzer import LLMAnalyzer
from pipelline_ppo.ppo_dataset_codah import (
    create_codah_ppo_dataset,
    load_ppo_dataset_jsonl,
)
from utils.custom_chat_template import custom_apply_chat_template
from utils.general import print_gpu_info

print_gpu_info()


def seq_ppo(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # llm_analyzer = LLMAnalyzer(
    #     model_id=model_name, device="cuda"
    # )
    # dataset = Dataset.from_dict({"query": prompts})

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]" if "mistral" in getattr(tokenizer, "name_or_path", "").lower() else "<|pad|>"})
        # base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        # TODO: Check if using the eos token is necessary
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(sample):
        tokens = tokenizer(custom_apply_chat_template(sample["query"]))

        # Preserve raw text in "raw_data" while adding tokenized fields
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    # dataset = create_codah_ppo_dataset(base_model, tokenizer, subset=10)
    dataset = load_ppo_dataset_jsonl(
        file_path=str(Path("ppo_datasets/ppo_codah_dataset.jsonl"))
    )

    dataset = Dataset.from_list(dataset)

    train_dataset = dataset.map(tokenize, batched=False)

    base_model = get_peft_model(base_model, lora_config).to(device)

    # model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    # # Convert base model to PPO model with a value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    # # Manually copy the generation config from the base model
    # model.generation_config = GenerationConfig.from_model_config(base_model.config)
    wandb_mode = False

    ppo_config = PPOConfig(
        model_name=model_name,
        # num_ppo_epochs=4,  # Sets num ppo updates per batch
        # Change learning rate
        learning_rate=1e-6,
        target_kl=0.1,
        init_kl_coef=0.1,
        steps=10000,
        # kl_coef=0.1,
        batch_size=64,
        mini_batch_size=8,  # Set mini_batch_size
        gradient_accumulation_steps=4,  # Set gradient_accumulation_steps
        # output_dir="ppo_output",
        log_with="wandb" if wandb_mode else None,
    )

    # Ensure correct padding and formatting during training
    data_collator_ob = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def custom_data_collator(features):
        # Extract tokenized fields for padding
        tokenized_samples = [
            {k: v for k, v in f.items() if k in ["input_ids", "attention_mask"]}
            for f in features
        ]

        # Apply padding only on tokenized fields
        batch = data_collator_ob(tokenized_samples)

        # # Keep the original queries inside the batch
        # batch["query"] = [f["query"] for f in features]  # Retain the query field

        # Store non-tensor fields separately to prevent DataLoader from dropping them
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "query": [
                f["query"] for f in features
            ],  # Keep queries as a list of strings
        }

        return batch

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        # ref_model=None,
        # reward_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        # eval_dataset=None,
        data_collator=custom_data_collator,
        # dataloader_kwargs={"drop_last": False},
    )

    # It is recommended to use those parameters!
    generation_kwargs = {
        "min_length": -1,  # don't ignore the EOS token (see above)
        # "min_new_tokens": 32,
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
        "do_sample": True,  # yes, we want to sample
        "pad_token_id": tokenizer.pad_token_id,
        # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 200,  # specify how many tokens you want to generate at most
    }

    # TODO: Think of whether to use the LLMAnalyzer (online PPO or not)
    llm_analyzer = LLMAnalyzer(model_id=model, tokenizer=tokenizer, device="cuda")
    print(llm_analyzer)

    mean_scores = []
    std_scores = []
    KL_scores = []
    avg_rewards = []
    num_epochs = 1  # Replace with the desired number of epochs

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch + 1}"):
            query_tensors = batch["input_ids"].to(device)

            query_tensors_list = [query.to(device) for query in query_tensors]

            # Generate responses for each query tensor
            response_tensors = ppo_trainer.generate(
                query_tensors_list, **generation_kwargs, return_prompt=False
            )

            response_tensors_list = [
                response.to(device) for response in response_tensors
            ]

            # Decode both the query and the response, filtering out padding tokens
            decoded_queries = [
                tokenizer.decode(q, skip_special_tokens=True).strip()
                for q in query_tensors_list
            ]

            decoded_responses = [
                tokenizer.decode(r, skip_special_tokens=True).strip()
                for r in response_tensors_list
            ]

            # Compute rewards
            # Change here to my method
            # rewards = pre.compute_reward_seq(
            #     decoded_responses, topic_model, relevant_topics
            # )
            rewards = [1 for _ in range(len(decoded_responses))]
            avg_r = sum(rewards) / len(rewards)
            for i in range(5):
                print(
                    "\n{}) {} --- {}".format(
                        i + 1, decoded_queries[i], decoded_responses[i]
                    )
                )

            rewards_tensor_list = [
                torch.tensor(r, dtype=torch.float32) for r in rewards
            ]

            # Run PPO step with KL control
            stats = ppo_trainer.step(
                query_tensors_list, response_tensors_list, rewards_tensor_list
            )

            mean_scores.append(round(stats["ppo/mean_scores"], 3))
            std_scores.append(round(stats["ppo/std_scores"], 3))
            KL_scores.append(round(stats["objective/kl"], 3))

            print("\nThe mean scores so far {}".format(mean_scores))
            print("\nThe STD scores so far {}".format(std_scores))
            # KL scores must be positive and as small as possible!
            print("\nThe KL scores so far {}".format(KL_scores))

    ppo_trainer.save_pretrained("my_ppo_model")
    with open("stats.pkl", "wb") as f:
        pickle.dump([mean_scores, std_scores, KL_scores, avg_rewards], f)


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    seq_ppo(model_name)
