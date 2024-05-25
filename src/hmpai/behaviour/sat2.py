from pathlib import Path
import pandas as pd


def read_behavioural_info(path: Path):
    data = pd.read_csv(path)
    
    # Remove indexing columns
    data = data.drop([data.columns[0], data.columns[8]], axis=1)
    return data