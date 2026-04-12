#!/usr/bin/env python3
"""
Comprehensive test script for ConstLLM environment
Run this after creating your conda environment to verify everything works.
"""

import sys
import traceback
from typing import Dict, List, Tuple


def test_basic_python():
    """Test basic Python and system info"""
    print("🔍 Testing Basic Python Setup...")
    try:
        print(f"✅ Python version: {sys.version}")
        print(f"✅ Python executable: {sys.executable}")
        return True
    except Exception as e:
        print(f"❌ Basic Python test failed: {e}")
        return False


def test_conda_packages():
    """Test conda-installed packages"""
    print("\n🔍 Testing Conda Packages...")
    conda_packages = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", "sklearn"),
        ("matplotlib.pyplot", "plt"),
        ("tqdm", "tqdm"),
    ]

    results = []
    for package, alias in conda_packages:
        try:
            exec(f"import {package} as {alias}")
            version = eval(f"{alias}.__version__") if hasattr(eval(alias), "__version__") else "unknown"
            print(f"✅ {package}: {version}")
            results.append(True)
        except Exception as e:
            print(f"❌ {package}: {e}")
            results.append(False)

    return all(results)


def test_pytorch():
    """Test PyTorch and CUDA"""
    print("\n🔍 Testing PyTorch and CUDA...")
    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")

            # Test GPU computation
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print(f"✅ GPU computation works, result shape: {z.shape}")
            print(f"✅ GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
            torch.cuda.empty_cache()
        else:
            print("⚠️  No CUDA available - running on CPU only")

        # Test torchvision and torchaudio
        import torchaudio
        import torchvision

        print(f"✅ Torchvision version: {torchvision.__version__}")
        print(f"✅ Torchaudio version: {torchaudio.__version__}")

        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        traceback.print_exc()
        return False


def test_ml_packages():
    """Test ML/NLP packages"""
    print("\n🔍 Testing ML/NLP Packages...")
    ml_packages = [
        "transformers",
        "datasets",
        "accelerate",
        "captum",
        "trl",
        "peft",
    ]

    results = []
    for package in ml_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {package}: {version}")
            results.append(True)
        except Exception as e:
            print(f"❌ {package}: {e}")
            results.append(False)

    return all(results)


def test_pip_packages():
    """Test pip-installed packages"""
    print("\n🔍 Testing Pip Packages...")
    pip_packages = [
        "bitsandbytes",
        "wandb",
        "unsloth",
    ]

    results = []
    for package in pip_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {package}: {version}")
            results.append(True)
        except Exception as e:
            print(f"❌ {package}: {e}")
            results.append(False)

    return all(results)


def test_model_loading():
    """Test actual model loading and inference"""
    print("\n🔍 Testing Model Loading and Inference...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "gpt2"  # Small model for testing
        print(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ Model and tokenizer loaded")

        # Test tokenization
        text = "Hello, this is a test."
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        print(f"✅ Tokenization works, input shape: {inputs['input_ids'].shape}")

        # Test inference
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✅ Model inference works, output shape: {outputs.logits.shape}")

        # Test generation
        generated = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 5,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✅ Text generation works: '{generated_text}'")

        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        traceback.print_exc()
        return False


def test_jupyter():
    """Test Jupyter functionality"""
    print("\n🔍 Testing Jupyter...")
    try:
        import IPython
        import jupyter

        print(f"✅ IPython version: {IPython.__version__}")
        print("✅ Jupyter available")
        return True
    except Exception as e:
        print(f"❌ Jupyter test failed: {e}")
        return False


def test_your_project():
    """Test your specific project imports"""
    print("\n🔍 Testing Your Project Imports...")
    try:
        # Add your project path - adjust as needed
        project_paths = ["./src", "."]

        for path in project_paths:
            if path not in sys.path:
                sys.path.append(path)

        # Test your specific imports
        from llm_attribution.LLMAnalyzer import LLMAnalyzer
        from llm_attribution.utils_attribution import AttributionMethod
        from utils.data_models import ScenarioItem

        print("✅ Your project modules import successfully")

        # Test enum
        print(f"✅ Attribution methods available: {[m.value for m in AttributionMethod]}")

        return True
    except Exception as e:
        print(f"⚠️  Your project imports failed (this is OK if you haven't set up the path): {e}")
        return True  # Don't fail the overall test for this


def main():
    """Run all tests"""
    print("🚀 Starting ConstLLM Environment Tests")
    print("=" * 50)

    tests = [
        ("Basic Python", test_basic_python),
        ("Conda Packages", test_conda_packages),
        ("PyTorch & CUDA", test_pytorch),
        ("ML/NLP Packages", test_ml_packages),
        ("Pip Packages", test_pip_packages),
        ("Model Loading", test_model_loading),
        ("Jupyter", test_jupyter),
        ("Your Project", test_your_project),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your environment is ready to use!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
