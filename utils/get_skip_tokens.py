from typing import List, Dict
import string
import nltk
from utils.ModelTokenizerBundle import ModelTokenizerBundle

def get_skip_tokens(
    tokenizer, extra_skip_tokens: list = None, only_skip_structure: bool = False
) -> dict:
    """
    Identifies stop words and punctuation tokens in the tokenizer's vocabulary.

    :param only_skip_structure:
    :param extra_skip_tokens:
    :param tokenizer: The Llama tokenizer instance.
    :return: A dictionary with two keys: "stop_words" and "punctuation",
             each containing a list of tokens.
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
        stripped_token = token.lstrip("ĊĠ▁").rstrip(
            "ĊĠ▁"
        )  # Remove leading/trailing symbols

        # Check if the stripped token is a stop word
        if stripped_token in stop_words:
            stop_word_tokens[token] = token_id

        # Check if the stripped token is a punctuation mark
        if stripped_token in punctuation:
            punctuation_tokens[token] = token_id

    structure_tokens = {
        "<|start_header_id|>": tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
        "<|end_header_id|>": tokenizer.convert_tokens_to_ids("<|end_header_id|>"),
        "<|eot_id|>": tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        "<|begin_of_text|>": tokenizer.convert_tokens_to_ids("<|begin_of_text|>"),
        "system": tokenizer.convert_tokens_to_ids("system"),
        "user": tokenizer.convert_tokens_to_ids("user"),
        "assistant": tokenizer.convert_tokens_to_ids("assistant"),
        "Ċ": tokenizer.convert_tokens_to_ids("Ċ"),
        "Ġ->": tokenizer.convert_tokens_to_ids("Ġ->"),
    }

    # Handle extra skip tokens
    extra_tokens_dict = {}
    if extra_skip_tokens:
        for extra_token in extra_skip_tokens:
            # Tokenize the string to decompose it into individual tokens
            tokenized_ids = tokenizer.encode(extra_token, add_special_tokens=False)

            # Add each token to the extra_tokens_dict
            for idx, token_id in enumerate(tokenized_ids):
                token_string = tokenizer.decode([token_id]).strip()
                extra_tokens_dict[f"{token_string}"] = token_id

    # Combine results into a single dictionary
    if only_skip_structure:
        skip_tokens_dict = structure_tokens
    else:
        skip_tokens_dict = {
            **stop_word_tokens,
            **punctuation_tokens,
            **structure_tokens,
            **extra_tokens_dict,
        }

    # Print results
    print("Total Tokens Skipped:", len(skip_tokens_dict))
    return skip_tokens_dict


# def get_structure_tokens(tokenizer):
#     structure_tokens = {
#         '<|start_header_id|>': tokenizer.convert_tokens_to_ids('<|start_header_id|>'),
#         '<|end_header_id|>': tokenizer.convert_tokens_to_ids('<|end_header_id|>'),
#         '<|eot_id|>': tokenizer.convert_tokens_to_ids('<|eot_id|>'),
#         '<|begin_of_text|>': tokenizer.convert_tokens_to_ids('<|begin_of_text|>'),
#         'system': tokenizer.convert_tokens_to_ids('system'),
#         'user': tokenizer.convert_tokens_to_ids('user'),
#         'assistant': tokenizer.convert_tokens_to_ids('assistant'),
#         # 'Ċ': tokenizer.convert_tokens_to_ids('Ċ'),
#     }
#
#     return structure_tokens