import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from captum.attr import (
    LayerIntegratedGradients,
    Lime,
    LLMGradientAttribution,
    ShapleyValueSampling,
    TextTokenInput,
)
from captum.attr._core.llm_attr import LLMAttributionResult

from llm_attribution.LLMAttributionMod import ExtendedLLMAttribution
from llm_attribution.TextTokenInputMod import TextTokenInputMod
from llm_attribution.utils_attribution import AttributionMethod
from utils.cutom_chat_template import custom_apply_chat_template
from utils.data_models import LLMAnalysisRes
from utils.get_skip_tokens import get_skip_tokens
from utils.ModelTokenizerBundle import ModelTokenizerBundle


class LLMAnalyzer:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        extra_skip_tokens: list[str] = None,
        only_structure_tokens: bool = True,
    ):
        # Load the tokenizer and model directly
        use_quantization = True
        model_bundle: ModelTokenizerBundle = ModelTokenizerBundle(
            model_id=model_id, use_quantization=use_quantization, device=device
        )
        self.tokenizer = model_bundle.tokenizer
        self.model = model_bundle.model
        self.model.eval()

        self.skip_tokens_dict = get_skip_tokens(
            tokenizer=self.tokenizer,
            extra_skip_tokens=extra_skip_tokens,
            only_skip_structure=only_structure_tokens,
        )
        self.embedding_layer = self._get_embedding_layer()

    def _get_embedding_layer(self):
        try:
            return self.model.transformer.wte
        except AttributeError:
            try:
                return self.model.model.embed_tokens
            except AttributeError:
                raise AttributeError("Cannot find the embedding layer in the model.")

    def _get_seq_attr_list(
        self,
        attribution_result: LLMAttributionResult,
    ) -> List[Tuple[str, float]]:
        """
        Converts an attribution result into a sequence of (token, score) tuples.

        :param attribution_result: The attribution result from the analyzer.
        :return: List of (token, score) tuples.
        """
        input_tokens = attribution_result.input_tokens
        seq_attr = attribution_result.seq_attr.cpu().tolist()

        # Ensure the lengths match
        if len(input_tokens) != len(seq_attr):
            raise ValueError(
                f"Shape mismatch: input_tokens has length {len(input_tokens)}, "
                f"but seq_attr has length {len(seq_attr)}."
            )
        return list(zip(input_tokens, seq_attr))

    def generate_output(self, input_text: str) -> str:
        # Ensure pad_token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set pad_token_id in model config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Manage cache warning
        self.model.generation_config.use_cache = False

        # Tokenize and generate
        inputs = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        return generated_text

    def _prepare_input(
        self, full_text: str, static_texts: Union[str, List[str], None] = None
    ) -> TextTokenInputMod:
        """
        Prepares the interpretable input with static and dynamic parts.

        Args:
            full_text (str): The full input text for the model.

        Returns:
            TextTokenInputWithStatic: Interpretable input with static tokens marked.
        """

        input_with_static = TextTokenInputMod(
            text=full_text,
            tokenizer=self.tokenizer,
            static_texts=static_texts,  # Let the class tokenize this internally
            skip_tokens=list(self.skip_tokens_dict.values()),
        )

        return input_with_static

    def analyze_lime(
        self,
        input_text: str,
        static_texts: Union[str, List[str], None],
        target: str,
        **params,
    ):
        interpretable_input = self._prepare_input(input_text, static_texts)
        attribution_method = Lime(self.model)
        # Check if 'baselines' is in params, and add it if missing
        params.setdefault("baselines", self.tokenizer.pad_token_id)

        llm_attr = ExtendedLLMAttribution(
            attr_method=attribution_method,
            tokenizer=self.tokenizer,
        )
        return llm_attr.attribute(
            inp=interpretable_input, target=target, show_progress=True, **params
        )

    def analyze_shapley_value_sampling(
        self, input_text: str, generated_output: str, **params
    ):
        interpretable_input = self._prepare_input(input_text)
        attribution_method = ShapleyValueSampling(self.model)
        # Check if 'baselines' is in params, and add it if missing
        params.setdefault("baselines", self.tokenizer.pad_token_id)
        shapley_attr = ExtendedLLMAttribution(
            attr_method=attribution_method,
            tokenizer=self.tokenizer,
        )
        return shapley_attr.attribute(
            inp=interpretable_input,
            target=generated_output,
            show_progress=True,
            **params,
        )

    def layer_integrated_gradients(
        self, input_text: str, generated_output: str, **params
    ):
        # Wrap the model in DataParallel for Captum
        # if torch.cuda.device_count() > 1:
        #     parallel_model = torch.nn.DataParallel(self.model)
        # else:
        #     parallel_model = self.model

        # Prepare interpretable inputs
        interpretable_input = self._prepare_input(input_text)

        # Define the Layer Integrated Gradients algorithm
        lig_method = LayerIntegratedGradients(
            forward_func=self.model, layer=self.embedding_layer
        )

        # Wrap LIG with LLMGradientAttribution for consistent processing
        gradient_attributor = LLMGradientAttribution(
            attr_method=lig_method, tokenizer=self.tokenizer
        )

        # Compute attribution
        return gradient_attributor.attribute(
            inp=interpretable_input,
            target=generated_output,
            **params,  # Pass additional params (e.g., n_steps, baselines)
        )

    # def layer_integrated_gradients(self, input_text: str, generated_output: str, **params):
    #     interpretable_input = self._prepare_input(input_text)
    #     attribution_method = LayerIntegratedGradients(self.model)
    #     layer_integrated_attr = ExtendedLLMAttribution(attr_method=attribution_method,
    #                                           tokenizer=self.tokenizer,
    #                                           )
    #     return layer_integrated_attr.attribute(inp=interpretable_input, target=generated_output, show_progress=True, **params)

    def analyze(
        self,
        input_text: str,
        static_texts: Union[str, List[str], None] = None,
        target: Optional[str] = None,
        method_params: Optional[
            Dict[AttributionMethod, Dict[str, LLMAttributionResult]]
        ] = None,
    ) -> LLMAnalysisRes:
        """
        Analyze input using attribution methods specified in method_params, where each method can have its own parameters.
        """
        method_params = method_params or {}

        # Generate output based on input text
        if target is None:
            target = self.generate_output(input_text)

        results = {}
        for method, params in method_params.items():
            print(f"Running {method}...")
            try:
                if method == AttributionMethod.LIME.name:
                    results[method] = self.analyze_lime(
                        input_text=input_text,
                        static_texts=static_texts,
                        target=target,
                        **params,
                    )
                elif method == AttributionMethod.SHAPLEY_VALUE_SAMPLING.name:
                    results[method] = self.analyze_shapley_value_sampling(
                        input_text, target, **params
                    )
                elif method == AttributionMethod.LAYER_INTEGRATED_GRADIENTS.name:
                    results[method] = self.layer_integrated_gradients(
                        input_text, target, **params
                    )
                # Add other methods here as needed
                else:
                    print(f"Method {method} is not supported.")

            except Exception as e:
                print(f"Error running {method}: {str(e)}")
                traceback.print_exc()
                continue

        methods_scores = {
            method: self._get_seq_attr_list(result)
            for method, result in results.items()
        }

        analysis_results = LLMAnalysisRes(
            input_text=input_text, target=target, methods_scores=methods_scores
        )
        return analysis_results


def run_mod_captum():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    analyzer = LLMAnalyzer(model_id=model_id)
    messages = []
    user_prompt_decision = """I am feeling nervous about my midterm tomorrow. I fear that
Choose the most plausible answer, respond only with the answer and the description:
A) the professor will delay the midterm.
B) I will doodle on my exam and receive points for it.
C) my grandpa has diabetes.
D) I will fail.
"""
    messages.append({"role": "user", "content": user_prompt_decision})
    decision_prompt = custom_apply_chat_template(messages)
    generated_output = analyzer.generate_output(decision_prompt)
    print(f"Generated output Check: {generated_output}")
    # method_params = {AttributionMethod.LIME: {'n_samples': 2000,
    #                                           'baselines': analyzer.tokenizer.pad_token_id,
    #                                           "perturbations_per_eval": 100},
    #                  # AttributionMethod.SHAPLEY_VALUE_SAMPLING: {"n_samples": 25,
    #                  #                                                 "baselines": analyzer.tokenizer.pad_token_id,
    #                  #                                                 "perturbations_per_eval": 1,}
    #                  }

    # lime_results = analyzer.analyze_lime(
    #     input_text=decision_prompt,
    #     static_texts=["Choose the most plausible answer:\n",
    #                   "Respond only with 'A', 'B', 'C', or 'D'."],
    #     generated_output=generated_output,
    #     **method_params[AttributionMethod.LIME]
    # )
    # sorted_all_tokens = sorted(
    #     lime_results.seq_attr_dict.items(),
    #     key=lambda x: abs(x[1]),
    #     reverse=True
    # )[:50]
    # print("All tokens (sorted by absolute value):")
    # for token, value in sorted_all_tokens:
    #     print(f"{token}: {value}")

    # try:
    #     lime_results = analyzer.analyze_lime(
    #         input_text=decision_prompt,
    #         generated_output=generated_output,
    #         token_classes=token_classes,
    #         **method_params[AttributionMethod.LIME]
    #     )
    #     sorted_token_classes = sorted(
    #         lime_results.seq_attr_dict.items(),
    #         key=lambda x: abs(x[1]),
    #         reverse=True
    #     )[:15]
    #     print("Only token classes (sorted by absolute value):")
    #     for token, value in sorted_token_classes:
    #         print(f"{token}: {value}")
    # except Exception as e:
    #     print(f"Error: {e}")

    # shapley_results = analyzer.analyze_shapley_value_sampling(
    #     input_text=decision_prompt,
    #     generated_output=generated_output,
    #     **method_params[AttributionMethod.SHAPLEY_VALUE_SAMPLING]
    # )
    # sorted_all_tokens = sorted(
    #     shapley_results.seq_attr_dict.items(),
    #     key=lambda x: abs(x[1]),
    #     reverse=True
    # )[:15]
    # print("All tokens (sorted by absolute value):")
    # for token, value in sorted_all_tokens:
    #     print(f"{token}: {value}")
    #
    #
    # try:
    #     shapley_results = analyzer.analyze_shapley_value_sampling(
    #         input_text=decision_prompt,
    #         generated_output=generated_output,
    #         token_classes=token_classes,
    #         **method_params[AttributionMethod.SHAPLEY_VALUE_SAMPLING]
    #     )
    #     sorted_token_classes = sorted(
    #         shapley_results.seq_attr_dict.items(),
    #         key=lambda x: abs(x[1]),
    #         reverse=True
    #     )[:15]
    #     print("Only token classes (sorted by absolute value):")
    #     for token, value in sorted_token_classes:
    #         print(f"{token}: {value}")
    # except Exception as e:
    #     print(f"Error: {e}")


if __name__ == "__main__":
    run_mod_captum()
