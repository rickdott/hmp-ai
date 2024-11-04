from pathlib import Path
import pandas as pd

SAT2_SPLITS = [
    ["S6", "S2", "S16", "S18", "S12", "S9", "S13", "S1", "S20"], # Train
    ["S5", "S11", "S15", "S17"], # Test
    ["S4", "S10", "S3", "S7", "S8"], # Val
]


def read_behavioural_info(path: Path):
    data = pd.read_csv(path)

    # Remove indexing columns
    data = data.drop([data.columns[0], data.columns[8]], axis=1)
    return data
