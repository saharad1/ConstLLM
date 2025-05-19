from datasets import load_dataset

codah_dataset = load_dataset(path="ai2_arc", name="ARC-Challenge", split="all")
print(len(codah_dataset))

# # Load the ECQA dataset
# ecqa_dataset = load_dataset(path="jaredfern/ecqa", name="ecqa", split="all")
# print(len(ecqa_dataset))
