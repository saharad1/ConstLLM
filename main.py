import os
import sys
import time

import psutil
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# print("CUDA available:", torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
# print("Current GPU:", torch.cuda.current_device())


# # List their names
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# import bitsandbytes as bnb
# from transformers import AutoModelForCausalLM

# from datasets import Dataset, load_dataset

# # from bitsandbytes import load

# print(bnb.__version__)

# import captum

# print(captum.__version__)

# print(sys.path)
# from llm_attribution.LLMAnalyzer import LLMAnalyzer

# print(LLMAnalyzer)

# # model = AutoModelForCausalLM.from_pretrained(
# #     "meta-llama/Llama-2-7b-hf",
# #     load_in_8bit=True,  # Enable 8-bit quantization
# #     device_map="auto",
# # )
# #
# # print(model)


cpu_percent_per_core = psutil.cpu_percent(percpu=True, interval=1)
for i, usage in enumerate(cpu_percent_per_core):
    print(f"Core {i}: {usage}% utilization")


for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
    print(proc.info)


torch.set_num_threads(1)  # Baseline single-threaded
start = time.time()
# Run Shapley attribution or some heavy tensor operation here
end = time.time()
print(f"Execution time with 1 thread: {end - start:.4f} seconds")

torch.set_num_threads(8)  # Multi-threaded
start = time.time()
# Run the same operation again
end = time.time()
print(f"Execution time with 8 threads: {end - start:.4f} seconds")
