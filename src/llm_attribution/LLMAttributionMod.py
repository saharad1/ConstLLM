from captum.attr import Lime, LLMAttribution, LLMAttributionResult, Attribution
from torch.nn import functional as F
from typing import Union, Callable, Dict, Optional, List
import torch


class ExtendedLLMAttribution(LLMAttribution):
    """
    Extended LLMAttribution class to support:
    1. Analysis on specific token classes.
    2. Optimized batch processing for runtime improvements.
    """

    def __init__(
        self,
        attr_method: Attribution,
        tokenizer,
        attr_target: str = "log_prob",  # ["log_prob", "prob"]
    ):
        """
        Initialize the extended attribution class.

        Args:
            attr_method (Lime): Instance of Captum's Lime attribution method.
            tokenizer: Tokenizer for the model.
            attr_target (str): Whether to attribute towards "log_prob" or "prob".
        """
        super().__init__(attr_method, tokenizer, attr_target)

    def _forward_func(
        self,
        perturbed_tensor: torch.Tensor,
        inp,
        target_tokens: Union[torch.Tensor, List[int]],
        _inspect_forward: Optional[Callable] = None,
    ):
        """
        Forward function optimized for batch processing and specific token classes.
        """
        perturbed_input_batch = self._format_model_input(
            inp.to_model_input(perturbed_tensor)
        )

        # Run model forward pass for the entire batch
        output_logits = self.model.forward(
            perturbed_input_batch,
            attention_mask=torch.ones_like(perturbed_input_batch).to(self.device),
        )

        # Get logits for the last token in each sequence
        new_token_logits = output_logits.logits[
            :, -1, :
        ]  # Shape: (batch_size, vocab_size)

        # Compute log probabilities for specific token classes or all tokens
        log_probs = F.log_softmax(new_token_logits, dim=-1)

        # Focus on specific token classes if provided
        relevant_log_probs = log_probs[:, target_tokens]  # Default behavior

        # Sum log probabilities across token classes if applicable
        total_log_prob = relevant_log_probs.sum(dim=1)

        # Optionally inspect forward results
        if _inspect_forward:
            prompt = self.tokenizer.decode(perturbed_tensor[0])
            response_tokens = [
                self.tokenizer.decode([token]) for token in target_tokens
            ]
            _inspect_forward(prompt, response_tokens, total_log_prob.tolist())

        return (
            total_log_prob.unsqueeze(1)
            if self.attr_target == "log_prob"
            else torch.exp(total_log_prob).unsqueeze(1)
        )

    def attribute(
        self,
        inp,
        target: Union[str, torch.Tensor, None] = None,
        num_trials: int = 1,
        gen_args: Optional[Dict] = None,
        **kwargs,
    ):
        self.include_per_token_attr = False
        # Generate target tokens if not provided
        if target is None:
            assert hasattr(self.model, "generate") and callable(
                self.model.generate
            ), "Model must have a generate method when target is not provided."
            if not gen_args:
                gen_args = {"max_length": 25, "do_sample": False}
            model_inp = self._format_model_input(inp.to_model_input())
            output_tokens = self.model.generate(model_inp, **gen_args)
            target_tokens = output_tokens[0][model_inp.size(1) :]
        else:
            if isinstance(target, str):
                target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
            elif isinstance(target, torch.Tensor):
                target_tokens = target.tolist()
            else:
                raise ValueError("Target must be a string or tensor.")

        # Initialize attribution results
        attr = torch.zeros(
            [
                1 + len(target_tokens) if self.include_per_token_attr else 1,
                inp.n_itp_features,
            ],
            dtype=torch.float,
            device=self.device,
        )

        # Perform attribution for each trial
        for _ in range(num_trials):
            attr_input = inp.to_tensor().to(self.device)
            cur_attr = self.attr_method.attribute(
                attr_input,
                additional_forward_args=(inp, target_tokens, None),
                **kwargs,
            )

            if cur_attr.dim() == 3 and cur_attr.size(0) == 1:
                # Remove batch dimension for consistency
                cur_attr = cur_attr.squeeze(0)
            elif cur_attr.dim() == 3:
                # Handle cases where multiple batches are processed together
                cur_attr = cur_attr.mean(dim=0)

            # Ensure cur_attr matches the expected shape
            if cur_attr.shape != attr.shape:
                raise ValueError(
                    f"Shape mismatch: cur_attr shape {cur_attr.shape} does not match attr shape {attr.shape}"
                )

            # Accumulate attributions
            attr += cur_attr

        # Average results across trials
        attr /= num_trials

        # Format attributions
        attr = inp.format_attr(attr)

        # Return as LLMAttributionResult for consistency with original behavior
        return LLMAttributionResult(
            seq_attr=attr[0],
            token_attr=attr[1:] if self.include_per_token_attr else None,
            input_tokens=inp.values,
            output_tokens=self.tokenizer.convert_ids_to_tokens(target_tokens),
        )
