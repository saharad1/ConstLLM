#!/usr/bin/env python3

import os

import torch


def test_gpu_setup():
    print("=== GPU Setup Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")

    print(f"\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")


def test_auto_device_mapping():
    print("\n=== Testing Auto Device Mapping ===")

    # Test with a small model first
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "microsoft/DialoGPT-small"  # Small model for testing

    print(f"Loading {model_id} with auto device mapping...")

    try:
        # Load with auto device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Model loaded successfully!")

        # Check device placement
        print("\nDevice placement:")
        for name, param in model.named_parameters():
            if param.device.type == "cuda":
                print(f"  {name}: {param.device}")
                break  # Just show first few

        # Test generation
        print("\nTesting generation...")
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, max_length=50, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {response}")

        # Check memory usage
        print("\nMemory usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_gpu_setup()
    test_auto_device_mapping()
