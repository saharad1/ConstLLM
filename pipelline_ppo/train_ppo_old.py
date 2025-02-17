from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import get_peft_model, LoraConfig, TaskType
import torch
import pre
from datasets import Dataset
from tqdm import tqdm
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import pickle



def seq_ppo(model_name, prompts, topic_model, relevant_topics):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Dataset.from_dict({"query": prompts})

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)




    def tokenize(sample):
        tokens = tokenizer(sample["query"])
        sample["input_ids"] = tokens["input_ids"]
        sample["attention_mask"] = tokens["attention_mask"]

        return sample

    dataset = dataset.map(tokenize, batched=False, remove_columns=["query"])


    print(dataset)


    model = get_peft_model(model, lora_config).to(device)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    ppo_config = PPOConfig(
        model_name=model_name,
        # kl_penaly may not be required
        kl_penalty="full", 
        steps=10000,
        # Change learning rate
        learning_rate=1e-6,
        target_kl=0.1,
        init_kl_coef=0.1,
        batch_size=100,
        mini_batch_size=25,  # Set mini_batch_size
        gradient_accumulation_steps=4,  # Set gradient_accumulation_steps
    )


    # Ensure correct padding and formatting during training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator

    )

    # It is recommended to use those parameters!
    generation_kwargs = {
        "min_length": -1,  # don't ignore the EOS token (see above)
        #"min_new_tokens": 32,
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
        "do_sample": True,  # yes, we want to sample
        "pad_token_id": tokenizer.pad_token_id,
        # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 32,  # specify how many tokens you want to generate at most
    }

    
    mean_scores = []
    std_scores = []
    KL_scores = []
    avg_rewards = []
    num_epochs = 10  # Replace with the desired number of epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch + 1}"):
            query_tensors = batch["input_ids"].to(device)

            query_tensors_list = [query.to(device) for query in query_tensors]

    
            # Generate responses for each query tensor
            response_tensors = ppo_trainer.generate(query_tensors_list, ** generation_kwargs, return_prompt=False)

            response_tensors_list = [response.to(device) for response in response_tensors]


            # Decode both the query and the response, filtering out padding tokens
            decoded_queries = [
                tokenizer.decode(q, skip_special_tokens=True).strip() for q in query_tensors_list
            ]

            decoded_responses = [
                tokenizer.decode(r, skip_special_tokens=True).strip() for r in response_tensors_list
            ]



            # Compute rewards
            # Change here to my method
            rewards = pre.compute_reward_seq(decoded_responses, topic_model, relevant_topics)
            avg_r = sum(rewards) / len(rewards)
            for i in range(5):
                print("\n{}) {} --- {}".format(i + 1, decoded_queries[i], decoded_responses[i]))

            rewards_tensor_list = [torch.tensor(r, dtype=torch.float32) for r in rewards]



            # Run PPO step with KL control
            stats = ppo_trainer.step(query_tensors_list, response_tensors_list, rewards_tensor_list)

            mean_scores.append(round(stats["ppo/mean_scores"], 3))
            std_scores.append(round(stats["ppo/std_scores"], 3))
            KL_scores.append(round(stats["objective/kl"], 3))

            print("\nThe mean scores so far {}".format(mean_scores))
            print("\nThe STD scores so far {}".format(std_scores))
            # KL scores must be positive and as small as possible!!
            print("\nThe KL scores so far {}".format(KL_scores))






    ppo_trainer.save_pretrained("my_ppo_model")
    with open("stats.pkl", 'wb') as f:
        pickle.dump([mean_scores, std_scores, KL_scores, avg_rewards], f)

