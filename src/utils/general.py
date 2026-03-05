import os

import torch


def ensure_output_directory(path: str):
    os.makedirs(path, exist_ok=True)


def print_gpu_info():
    """Prints available GPUs, their names, and memory usage."""
    if not torch.cuda.is_available():
        print("No CUDA GPUs available")
        return

    device_count = torch.cuda.device_count()
    print(f"Available GPUs: {device_count}")

    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        avail, total = (x / 1024**3 for x in torch.cuda.mem_get_info(i))  # Convert to GB
        print(f"GPU {i}: {name} | Available: {avail:.2f} GB | Total: {total:.2f} GB")
