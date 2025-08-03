import torch
from datasets import load_dataset

# Check if PyTorch is installed and available
try:
    print(f"PyTorch version: {torch.__version__}")
except:
    print("PyTorch is not installed")


# Check if CUDA (GPU) is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Print number of available GPUs
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Print info for each GPU
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# codah_dataset = load_dataset(path="ai2_arc", name="ARC-Challenge", split="all")
# print(len(codah_dataset))

# # Load the ECQA dataset
# ecqa_dataset = load_dataset(path="jaredfern/ecqa", name="ecqa", split="all")
# print(len(ecqa_dataset))
