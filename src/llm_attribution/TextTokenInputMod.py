from typing import List, Union

import torch
from captum.attr import TextTokenInput


class TextTokenInputMod(TextTokenInput):
    """
    Extends TextTokenInput to handle multiple static texts (by positions) separately from skip_tokens.
    Static tokens in the defined ranges are excluded from attribution computation
    but included during generation.
    """

    def __init__(
        self,
        text: str,
        tokenizer,
        static_texts: Union[str, List[str], None] = None,  # Static portions of the text
        baselines: Union[int, str] = 0,  # Baseline token
        skip_tokens: Union[List[int], List[str], None] = None,  # Skip tokens
    ):
        """
        Args:
            text (str): The full input text for the model.
            tokenizer: Tokenizer of the model.
            static_texts (Union[str, List[str]]): Specific static portions of the text to exclude.
            baselines (Union[int, str]): Baseline token ID or token.
            skip_tokens (Union[List[int], List[str], None]): Tokens to skip entirely.
        """
        # Initialize parent class (handles skip_tokens logic)
        super().__init__(text, tokenizer, baselines=baselines, skip_tokens=skip_tokens)

        # Handle case when static_texts is None
        if static_texts is None:
            return  # No static texts to process, keep the default interpretable tensor

        # Ensure static_texts is a list
        if isinstance(static_texts, str):
            static_texts = [static_texts]

        # Process each static text and update the interpretable mask
        for static_text in static_texts:
            # Tokenize the input and identify static token positions
            inp_tensor = tokenizer.encode(text, return_tensors="pt")
            static_tokens = tokenizer.encode(static_text, add_special_tokens=False)

            input_tokens = inp_tensor[0].tolist()
            static_positions = []

            static_len = len(static_tokens)
            for start_idx in range(len(input_tokens) - static_len + 1):
                if input_tokens[start_idx : start_idx + static_len] == static_tokens:
                    static_positions.extend(range(start_idx, start_idx + static_len))
                    break
            else:
                raise ValueError(
                    f"Static text '{static_text}' does not match any part of the input text."
                )

            # Update the interpretable mask to exclude static tokens by positions
            self.itp_mask[0, static_positions] = False  # Exclude static token positions

        # Update the interpretable tensor after processing all static texts
        self.itp_tensor = self.inp_tensor[self.itp_mask].unsqueeze(0)

        # Update features and n_itp_features
        self.values = tokenizer.convert_ids_to_tokens(self.itp_tensor[0].tolist())
        self.n_itp_features = len(self.values)

        # Debugging Information
        # print(f"Static Texts: {static_texts}")
        # print(f"Interpretable Mask (itp_mask): {self.itp_mask}")
        # print(f"Interpretable Tensor (itp_tensor): {self.itp_tensor}")
