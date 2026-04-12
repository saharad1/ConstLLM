import string
from typing import Dict, List

import nltk

from src.utils.ModelTokenizerBundle import ModelTokenizerBundle


def get_skip_tokens(tokenizer, extra_skip_tokens: list = None, only_skip_structure: bool = True) -> dict:
    """
    Identifies stop words, punctuation, and model-specific structure tokens in the tokenizer's vocabulary.
    Works across different model families (Llama, Mistral, Phi, etc.)

    :param tokenizer: The model tokenizer instance
    :param extra_skip_tokens: Additional tokens to skip
    :param only_skip_structure: If True, only return structure tokens
    :return: A dictionary with token ids to skip during attribution
    """
    # Download necessary resources
    try:
        # Check if stopwords are cached
        nltk.data.find("corpora/stopwords")
        print("Stopwords found.")
    except LookupError:
        # Download stopwords if missing
        nltk.download("stopwords", quiet=True)
        print("Stopwords downloaded.")
    # nltk.download('stopwords')

    # Initialize stop words and punctuation
    stop_words = set(nltk.corpus.stopwords.words("english"))
    punctuation = set(string.punctuation)

    # Retrieve the tokenizer's vocabulary
    vocab = tokenizer.get_vocab()

    # Initialize containers for identified tokens
    stop_word_tokens = {}
    punctuation_tokens = {}

    # Analyze the tokenizer's vocabulary
    for token, token_id in vocab.items():
        # Normalize token to handle leading/trailing characters
        stripped_token = token.lstrip("ĊĠ▁").rstrip("ĊĠ▁")  # Remove leading/trailing symbols

        # Check if the stripped token is a stop word
        if stripped_token in stop_words:
            stop_word_tokens[token] = token_id

        # Check if the stripped token is a punctuation mark
        if stripped_token in punctuation:
            punctuation_tokens[token] = token_id

    # Detect model type and set appropriate structure tokens
    model_name = tokenizer.name_or_path.lower() if hasattr(tokenizer, "name_or_path") else ""
    structure_tokens = {}

    # Add basic special tokens that most models have
    for token_name in ["bos_token", "eos_token", "pad_token", "sep_token", "cls_token", "mask_token"]:
        token = getattr(tokenizer, token_name, None)
        if token is not None and token != "":
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                structure_tokens[token] = token_id
            except:
                pass

    # Try to add role-related tokens that most models use
    common_role_tokens = ["system", "user", "assistant"]
    for token in common_role_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                structure_tokens[token] = token_id
        except:
            pass

    # Add model-specific structure tokens
    if "mistral" in model_name:
        # Mistral-specific tokens
        mistral_tokens = ["<s>", "</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]
        for token in mistral_tokens:
            try:
                # Try direct token first
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    structure_tokens[token] = token_id
                else:
                    # If not found directly, try encoding and then getting individual token IDs
                    encoded = tokenizer.encode(token, add_special_tokens=False)
                    if encoded:
                        for idx, token_id in enumerate(encoded):
                            token_string = tokenizer.decode([token_id]).strip()
                            structure_tokens[f"{token_string}_{idx}"] = token_id
            except:
                # If that still fails, try to find tokens that might contain parts of these markers
                try:
                    for vocab_token, token_id in vocab.items():
                        if token in vocab_token:
                            structure_tokens[vocab_token] = token_id
                except:
                    pass

    elif "llama" in model_name:
        # Llama-specific tokens
        llama_tokens = [
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|begin_of_text|>",
            "Ċ",
            "Ġ->",
        ]
        for token in llama_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    structure_tokens[token] = token_id
            except:
                pass

    elif "phi" in model_name:
        # Phi-specific tokens based on actual observed tokens
        phi_tokens = [
            # Standard special tokens
            "<s>",
            "</s>",
            "<unk>",
            "<pad>",
            "<cls>",
            "<sep>",
            "<mask>",
            "<eod>",
            "<|user|>",
            "<|end|>",
            "<|endoftext|>",
            # Specific chat format tokens from our test output
            "assistant:",
            "user:",
            "system:",
        ]
        for token in phi_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    structure_tokens[token] = token_id
            except:
                pass

    elif "qwen" in model_name:
        # Qwen-specific tokens based on their chat format
        qwen_tokens = [
            # Standard special tokens
            "<s>",
            "</s>",
            "<unk>",
            "<pad>",
            # Qwen chat format tokens
            "<|im_start|>",
            "<|im_end|>",
            "system",
            "user",
            "assistant",
        ]
        for token in qwen_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    structure_tokens[token] = token_id
            except:
                pass

    # Combine the identified tokens
    skip_tokens = {}
    if not only_skip_structure:
        skip_tokens.update(stop_word_tokens)
        skip_tokens.update(punctuation_tokens)
    skip_tokens.update(structure_tokens)

    # Add extra skip tokens if provided
    if extra_skip_tokens is not None:
        for token in extra_skip_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                skip_tokens[token] = token_id
            except:
                pass

    return skip_tokens
