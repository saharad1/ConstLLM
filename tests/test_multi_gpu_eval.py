#!/usr/bin/env python3

import os

import torch

from src.llm_attribution.LLMAnalyzer import LLMAnalyzer


def test_multi_gpu_setup():
    print("=== Multi-GPU Evaluation Test ===")

    # Set up GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")

    # Test with a small model first
    model_id = "microsoft/DialoGPT-small"

    print(f"\nTesting LLMAnalyzer with device_map='auto'...")
    try:
        # Test with auto device mapping
        analyzer = LLMAnalyzer(model_id=model_id, device_map="auto", temperature=0.7)

        print("✓ LLMAnalyzer loaded successfully with auto device mapping!")

        # Test generation
        test_input = "Hello, how are you today?"
        print(f"\nTesting generation with input: '{test_input}'")

        output = analyzer.generate_output(test_input)
        print(f"Generated output: {output}")

        # Check memory usage
        print("\nMemory usage across GPUs:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        del analyzer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ Error with auto device mapping: {e}")
        return False

    return True


def test_with_llama_model():
    print("\n=== Testing with Llama-3.2-3B Model ===")

    # Test with the actual model you'll be using
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    print(f"Testing LLMAnalyzer with {model_id} and device_map='auto'...")
    try:
        analyzer = LLMAnalyzer(model_id=model_id, device_map="auto", temperature=0.7)

        print("✓ Llama model loaded successfully with auto device mapping!")

        # Test generation
        test_input = "What is the capital of France?"
        print(f"\nTesting generation with input: '{test_input}'")

        output = analyzer.generate_output(test_input)
        print(f"Generated output: {output}")

        # Check memory usage
        print("\nMemory usage across GPUs:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        del analyzer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ Error with Llama model: {e}")
        return False

    return True


if __name__ == "__main__":
    print("Testing multi-GPU evaluation setup...")

    # Test basic multi-GPU setup
    success1 = test_multi_gpu_setup()

    if success1:
        print("\n✓ Basic multi-GPU setup works!")

        # Test with actual model
        success2 = test_with_llama_model()

        if success2:
            print("\n✓ Multi-GPU evaluation is ready!")
            print("\nYou can now run your evaluation with:")
            print("./scripts/eval_trained_dpo_multi_gpu.sh")
        else:
            print("\n✗ Llama model test failed")
    else:
        print("\n✗ Basic multi-GPU setup failed")
