import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Model & Tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_responses(prompt, num_generations=3):
    """Generate multiple responses using the same sampling parameters."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    responses = []

    for _ in range(num_generations):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100, 
                do_sample=True,  # ENABLE SAMPLING
                temperature=0.8,  # Keep it the same
                top_p=0.9,        # Keep it the same
                top_k=40          # Keep it the same
            )
        responses.append(tokenizer.decode(output[0], skip_special_tokens=True))

    return responses

# Example Usage
prompt = "Explain the advantages of reinforcement learning in AI."
generated_responses = generate_responses(prompt, num_generations=3)

for idx, resp in enumerate(generated_responses):
    print(f"Response {idx + 1}:\n{resp}\n")