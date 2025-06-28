from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
import datasets
import pandas as pd
import os
import time
from huggingface_hub import login


def load_huggingface_dataset(
    dataset_id: str,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Load a dataset from Hugging Face. Whether the data is jsonl, csv, parquet or any other format, it will be loaded as a Hugging Face Dataset.
    Args:
        dataset_id (str): The ID of the dataset on Hugging Face.
    Returns:
        datasets.arrow_dataset.Dataset: The loaded dataset.
    """
    try:
        dataset = datasets.load_dataset(dataset_id)
        if isinstance(dataset, dict):
            # If the dataset is a dictionary, return the first split
            return list(dataset.values())[0]
        return dataset
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset from Hugging Face with ID '{dataset_id}': {e}"
        )


def setup_run_name(*, name: str | None, prefix: str = "", suffix: str = "") -> str:
    """
    Set up the run name for the training process.
    The run name is constructed from the base model ID, project name, and optional prefixes/suffixes.
    """
    # TODO: Pull the run name from the MLFLow
    if name is None:
        run_name = str(int(time.time()))  # in seconds since epoch
    else:
        if any(
            char in name for char in "$#@&*!"
        ):  # NOTE: Make this regex if getting more complex
            raise ValueError(
                "Run name contains invalid special characters: $, #, @, &, *, !"
            )
        run_name = name.strip()
    if prefix != "":
        run_name = prefix.strip() + "_" + run_name.strip()
    if suffix != "":
        run_name = run_name.strip() + "_" + suffix.strip()
    return run_name


def login_huggingface():
    """
    Log in to Hugging Face using the token from environment variables.
    Raises:
        ValueError: If the Hugging Face token is not set in the environment variables.
    """
    login(token=os.getenv("HF_TOKEN"))


def push_dataset_to_huggingface(
    repo_id: str, dataset_path: str
):
    """
    Push a dataset to HuggingFace Hub. If the dataset already exists, update it with a new commit.

    Args:
        repo_id (str): The repository ID on HuggingFace Hub (e.g., 'username/repo_name').
        dataset_path (str): Path to the dataset folder or file.
    """
    # HF support other types as well, but we are only supporting jsonl for now
    if dataset_path.endswith(".jsonl"):
        file_type = "json"
    else:
        raise NotImplementedError(
            "Unsupported dataset file type. Only .jsonl is supported yet."
        )
    # Convert the dataset to a HuggingFace Dataset
    dataset = load_dataset(
        file_type,
        data_files=dataset_path,
        split="train",
    )
    # Check if the dataset is an IterableDataset or IterableDatasetDict which dont support pushing
    if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
        raise NotImplementedError(
            "Pushing IterableDataset or IterableDatasetDict to HuggingFace Hub is not supported yet."
        )
    # Push the dataset to HuggingFace Hub
    dataset.push_to_hub(
        repo_id=repo_id,
        token=os.getenv("HF_TOKEN"),
        private=False,  # Set to True if you want the dataset to be private. INFO: Not interested RN
    )