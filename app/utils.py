from datasets import Dataset
import datasets
import pandas as pd
import os


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
    
        
def _convert_df_to_hf_dataset(self, df) -> datasets.arrow_dataset.Dataset:
    return Dataset.from_pandas(df)