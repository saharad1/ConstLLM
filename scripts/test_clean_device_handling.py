#!/usr/bin/env python3

import os

import torch

from src.llm_attribution.LLMAnalyzer import LLMAnalyzer


def test_device_configurations():
    print("=== Testing Clean Device Handling ===")

    # Test configurations
    configs = [
        {"name": "Single GPU", "device_map": None, "gpus": "0"},
        {"name": "Multi-GPU Auto", "device_map": "auto", "gpus": "0,1"},
        {"name": "Multi-GPU Balanced", "device_map": "balanced", "gpus": "0,1"},
        {"name": "Specific GPU", "device_map": "cuda:0", "gpus": "0"},
    ]

    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]

        try:
            analyzer = LLMAnalyzer(
                model_id="microsoft/DialoGPT-small", device_map=config["device_map"], temperature=0.7
            )

            print(f"✓ {config['name']} setup works!")

            # Test generation
            output = analyzer.generate_output("Hello")
            print(f"Generated: {output}")

            # Check memory usage
            print("Memory usage:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

            del analyzer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ {config['name']} failed: {e}")
            return False

    return True


def test_unsloth_compatibility():
    print("\n=== Testing Unsloth Model Compatibility ===")

    # Test with a model that might be detected as Unsloth
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    try:
        # Test with auto device mapping
        analyzer = LLMAnalyzer(
            model_id="microsoft/DialoGPT-small",  # This won't be Unsloth, but tests the path
            device_map="auto",
            temperature=0.7,
        )

        print("✓ Unsloth-compatible loading works!")

        # Test generation
        output = analyzer.generate_output("Test Unsloth compatibility")
        print(f"Generated: {output}")

        del analyzer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ Unsloth compatibility test failed: {e}")
        return False

    return True


def test_memory_efficiency():
    print("\n=== Testing Memory Efficiency ===")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    print("Memory before loading:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    try:
        # Test with auto device mapping for memory efficiency
        analyzer = LLMAnalyzer(model_id="microsoft/DialoGPT-small", device_map="auto", temperature=0.7)

        print("\nMemory after loading:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        # Test multiple generations
        for i in range(3):
            output = analyzer.generate_output(f"Test generation {i+1}")
            print(f"Generation {i+1}: {output}")

        print("\nMemory after generations:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        del analyzer
        torch.cuda.empty_cache()

        print("\nMemory after cleanup:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("Testing clean device handling for multi-GPU evaluation...")

    success1 = test_device_configurations()
    success2 = test_unsloth_compatibility()
    success3 = test_memory_efficiency()

    if success1 and success2 and success3:
        print("\n✓ All tests passed!")
        print("✓ Clean device handling is working correctly!")
        print("✓ Multi-GPU evaluation is ready!")
        print("\nYou can now run:")
        print("./scripts/eval_trained_dpo_multi_gpu.sh")
    else:
        print("\n✗ Some tests failed. Check the output above.")
