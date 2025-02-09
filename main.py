import torch
import sys
print("Sahar3")
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.current_device())

# List their names
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


from transformers import AutoModelForCausalLM
# from bitsandbytes import load

import bitsandbytes as bnb
print(bnb.__version__)

import captum
print(captum.__version__)

print(sys.path)
from llm_attribution.LLMAnalyzer import LLMAnalyzer
print(LLMAnalyzer)

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-7b-hf",
#     load_in_8bit=True,  # Enable 8-bit quantization
#     device_map="auto",
# )
#
# print(model)