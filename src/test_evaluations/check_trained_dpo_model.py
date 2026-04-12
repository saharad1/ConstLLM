from pathlib import Path

from src.llm_attribution.LLMAnalyzer import LLMAnalyzer
from src.utils.custom_chat_template import custom_apply_chat_template

# Example path to your trained DPO model
model_path = Path("trained_models/ecqa_models/LLama-instruct-8b/ecqa_spearman_250320_170948/final-model")

# Create analyzer with the model path
analyzer = LLMAnalyzer(model_id=str(model_path))

prompt = "What is the capital of France?"
# Example prompt
prompt_chat_template = custom_apply_chat_template(
    [
        {"role": "user", "content": prompt},
    ]
)

# Print attributions
print(analyzer.generate_output(prompt_chat_template))
