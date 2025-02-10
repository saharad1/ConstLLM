import torch


def print_memory_usage_all_gpus(message=""):
    """Prints the GPU memory usage for all available GPUs in MB"""
    print(f"\n{message}")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)  # Switch to GPU i
        allocated = torch.cuda.memory_allocated(i) / 1024**2  # Convert bytes to MB
        reserved = torch.cuda.memory_reserved(i) / 1024**2  # Convert bytes to MB
        print(f"GPU {i}: Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
