from typing import Dict, List

from src.utils.ModelTokenizerBundle import ModelTokenizerBundle


def custom_apply_chat_template(messages: List, add_generation_prompt=True, tokenize=False, tokenizer=None):
    """
    Uses the tokenizer's built-in apply_chat_template method but with customizations:
    - Makes it easier to use with an optional tokenizer parameter
    - Maintains the same interface as the previous custom implementation
    
    Args:
        messages (list of dict): Each dict has 'role' and 'content'.
        add_generation_prompt (bool): Whether to add the assistant's role for generation.
        tokenize (bool): Whether to return tokenized input.
        tokenizer: The tokenizer to use for tokenization.

    Returns:
        str or dict: The constructed prompt as a string, or tokenized input if tokenize=True.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    # Use the built-in method from the tokenizer
    formatted_chat = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
        return_tensors="pt" if tokenize else None
    )
    
    return formatted_chat


if __name__ == "__main__":
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_bundle = ModelTokenizerBundle(model_id=model_id, use_quantization=True)
    tokenizer = model_bundle.tokenizer
    # print(get_structure_tokens(tokenizer))
    # get_skip_tokens(tokenizer, extra_skip_tokens=["Premise", "Hypothesis"])
    # result = get_stop_and_punctuation_tokens(tokenizer)
    # print("Stop Words Tokens:", result["stop_words"])
    # print("Punctuation Tokens:", result["punctuation"])
    # stop_words = nlp.Defaults.stop_words
    # print(f"Number of stop words: {len(stop_words)}")
    # print(f"Sample stop words: {list(stop_words)[:10]}")

    # # Model and tokenizer initialization
    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_bundle = ModelTokenizerBundle(model_id=model_id, use_quantization=True)
    #
    # user_prompts = [
    #     """You are in a tiny room with no windows. Please respond with only 'A' or 'B', nothing else:
    #     A) Get out
    #     B) Stay
    #     Respond only with 'A' or 'B'.""",
    #     "Why did you make that choice? Explain briefly."
    # ]
    #
    # # Messages without system prompt
    # messages = [
    #     {"role": "user", "content": user_prompts[0]}
    # ]
    #
    # # Use the custom function to create the prompt
    # templated_input = custom_apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     tokenize=False,
    #     tokenizer=model_bundle.tokenizer
    # )
    #
    # print("Constructed Prompt:\n", templated_input)
    #
    # # Tokenize the prompt
    # model_input = model_bundle.tokenizer(
    #     templated_input,
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True,
    #     # pad_token=model_bundle.tokenizer.pad_token  # Use the tokenizer's pad token
    # ).to("cuda")
    #
    # # Generate the response
    # model_bundle.model.eval()
    # with torch.no_grad():
    #     output_ids = model_bundle.model.generate(
    #         model_input["input_ids"],
    #         max_new_tokens=30,
    #         pad_token_id=model_bundle.tokenizer.pad_token_id
    #     )[0]
    #
    #     # Decode the response (skip special tokens)
    #     response = model_bundle.tokenizer.decode(
    #         output_ids[len(model_input["input_ids"][0]):],
    #         skip_special_tokens=True
    #     )
    #     print("Model response:", response)
