from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
import datasets
import pandas as pd
import os
import time
from huggingface_hub import login

# DEPRECATED: Will be removed in future versions
def _load_local_dataset(self, data_dir):
    """
    Load a dataset from a local directory. Assumes there is one file in the directory.
    Returns a pandas DataFrame.
    """
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    if not files:
        raise FileNotFoundError(f"No data file found in {data_dir}")
    file_path = os.path.join(data_dir, files[0])
    self.logger.info(f"Loading data from {file_path}")
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
# DEPRECATED: Will be removed in future versions
def _convert_df_to_hf_dataset(self, df) -> datasets.arrow_dataset.Dataset:
    return Dataset.from_pandas(df)


def load_huggingface_dataset(dataset_id: str) -> (DatasetDict | Dataset | IterableDatasetDict | IterableDataset):
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
            raise ValueError(f"Failed to load dataset from Hugging Face with ID '{dataset_id}': {e}")
        
def setup_run_name(*, name: str | None, prefix: str = "", suffix: str = "") -> str:
    """
    Set up the run name for the training process.
    The run name is constructed from the base model ID, project name, and optional prefixes/suffixes.
    """
    # TODO: Pull the run name from the MLFLow
    if name is None:
        run_name = str(int(time.time()))   # in seconds since epoch
    else:
        if any(char in name for char in "$#@&*!"):  # NOTE: Make this regex if getting more complex
            raise ValueError("Run name contains invalid special characters: $, #, @, &, *, !")
        run_name = name.strip()
    run_name = prefix + run_name + suffix
    return run_name

def login_huggingface():
    """
    Log in to Hugging Face using the token from environment variables.
    Raises:
        ValueError: If the Hugging Face token is not set in the environment variables.
    """
    login(token=os.getenv("HF_TOKEN"))
