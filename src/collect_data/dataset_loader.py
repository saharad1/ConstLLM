# src/collect_data/dataset_loader.py
import logging

from datasets import load_dataset

from src.prepare_datasets.prepare_arc import PreparedARCDataset
from src.prepare_datasets.prepare_choice75 import PreparedCHOICE75Dataset
from src.prepare_datasets.prepare_codah import PreparedCODAHDataset
from src.prepare_datasets.prepare_ecqa import PreparedECQADataset

logger = logging.getLogger(__name__)


def load_and_prepare_dataset(dataset_name, subset=20):
    """
    Load and prepare a dataset by name.

    Args:
        dataset_name: Name of the dataset to load
        subset: Size of subset to use (if applicable)

    Returns:
        Prepared dataset object
    """
    logger.info(f"Loading and preparing dataset: {dataset_name}")

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

    logger.info(f"Number of scenarios: {len(prepared_dataset)}")
    return prepared_dataset
