import ast
import json


def fix_jsonl_file_advanced(input_file, output_file):
    """
    Fix a malformed JSONL file by safely evaluating Python literals
    and converting them to JSON.

    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSONL file
    """
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Skip empty lines
            if not line.strip():
                continue

            try:
                # Safely evaluate the Python literal
                data = ast.literal_eval(line)

                # Convert to JSON and write to output file
                json.dump(data, outfile)
                outfile.write("\n")
            except Exception as e:
                print(f"Error processing line: {e}")
                print(f"Problematic line: {line}")
                # Optionally, you can write the original line to the output file
                # outfile.write(line)


import json
from collections import Counter


def analyze_jsonl_file(jsonl_file, num_samples=5):
    """
    Analyze a JSONL file by examining the structure of the JSON objects.

    Args:
        jsonl_file: Path to the JSONL file to analyze
        num_samples: Number of samples to print
    """
    try:
        with open(jsonl_file, "r") as f:
            lines = f.readlines()

        print(f"Total lines in file: {len(lines)}")

        # Count the number of valid JSON lines
        valid_count = 0
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue

            try:
                json.loads(line)
                valid_count += 1
            except json.JSONDecodeError:
                pass

        print(f"Valid JSON lines: {valid_count}")

        # Analyze the structure of the first few valid JSON objects
        print("\nAnalyzing structure of the first few valid JSON objects:")
        count = 0
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                print(f"\nLine {i}:")
                print(f"Keys: {list(data.keys())}")

                # Check for the problematic fields
                if "decision_attributions" in data:
                    print(f"decision_attributions type: {type(data['decision_attributions'])}")
                    if data["decision_attributions"]:
                        print(f"First item type: {type(data['decision_attributions'][0])}")

                if "explanation_attributions" in data:
                    print(f"explanation_attributions type: {type(data['explanation_attributions'])}")
                    if data["explanation_attributions"]:
                        print(f"First item type: {type(data['explanation_attributions'][0])}")
                        if data["explanation_attributions"][0]:
                            print(f"First nested item type: {type(data['explanation_attributions'][0][0])}")

                count += 1
                if count >= num_samples:
                    break
            except json.JSONDecodeError:
                pass

    except Exception as e:
        print(f"Error analyzing file: {e}")


if __name__ == "__main__":
    # Fix the specific file that's causing the issue
    input_file = "data/collection_data/ecqa/meta-llama_Llama-3.2-3B-Instruct/ecqa_20250403_133655_LIG_llama3.2/ecqa_20250403_133655_LIG_llama3.2.jsonl"
    output_file = "data/collection_data/ecqa/meta-llama_Llama-3.2-3B-Instruct/ecqa_20250403_133655_LIG_llama3.2/ecqa_20250403_133655_LIG_llama3.2_fixed.jsonl"

    print(f"Fixing JSONL file: {input_file}")
    print(f"Output will be saved to: {output_file}")

    fix_jsonl_file_advanced(input_file, output_file)
    print(f"Fixed JSONL file saved to {output_file}")

    # Analyze the fixed file
    print("\nAnalyzing the fixed file:")
    analyze_jsonl_file(output_file)
