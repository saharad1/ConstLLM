"""
Shared utilities for dataset operations in the pipeline_dpo module.
"""

from typing import Union

from datasets import load_dataset

from src.prepare_datasets.prepare_arc import PreparedARCDataset
from src.prepare_datasets.prepare_choice75 import PreparedCHOICE75Dataset
from src.prepare_datasets.prepare_codah import PreparedCODAHDataset
from src.prepare_datasets.prepare_ecqa import PreparedECQADataset


def load_original_dataset(dataset_name: str, subset: Union[int, slice] = None):
    """
    Load the original dataset.

    Args:
        dataset_name: Name of the dataset to load
        subset: Optional subset size or slice to use

    Returns:
        Prepared dataset object
    """
    if dataset_name == "codah":
        raw_dataset = load_dataset(path="jaredfern/codah", name="codah", split="all")
        prepared_dataset = PreparedCODAHDataset(raw_dataset, subset=subset)
    elif dataset_name == "choice75":
        prepared_dataset = PreparedCHOICE75Dataset(subset=subset)
    elif dataset_name == "ecqa":
        raw_dataset = load_dataset(path="yangdong/ecqa", split="all")
        prepared_dataset = PreparedECQADataset(raw_dataset, subset=subset)
    elif dataset_name == "arc_easy":
        raw_dataset = load_dataset(path="ai2_arc", name="ARC-Easy", split="all")
        prepared_dataset = PreparedARCDataset(raw_dataset, subset=subset)
    elif dataset_name == "arc_challenge":
        raw_dataset = load_dataset(path="ai2_arc", name="ARC-Challenge", split="all")
        prepared_dataset = PreparedARCDataset(raw_dataset, subset=subset)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return prepared_dataset
