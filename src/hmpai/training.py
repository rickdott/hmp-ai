import random
from hmpai.data import preprocess
from sklearn.metrics import classification_report
import netCDF4
import xarray as xr
from typing import Callable
from hmpai.normalization import get_norm_vars, norm_dummy
from hmpai.pytorch.utilities import set_global_seed
import sklearn
import numpy as np
from copy import deepcopy
from hmpai.utilities import MASKING_VALUE
from collections import defaultdict
from pathlib import Path
import pandas as pd


def split_participants_into_folds(
    data_paths: list[str | Path],
    n_folds: int,
    shuffle: bool = True,
    participants_to_use: list[str] = None,
):
    participants = []
    for data_path in data_paths:
        with xr.open_dataset(data_path) as ds:
            participants.extend(ds.participant.values.tolist())

    if participants_to_use is not None:
        participants = [p for p in participants if p in participants_to_use]

    if n_folds > len(participants):
        raise ValueError(
            "Cannot provide more folds than participants, would result in an empty fold"
        )

    # If shuffling and still want to create predictability in folding, set seed before every call of this function
    if shuffle:
        np.random.shuffle(participants)

    return np.array_split(participants, n_folds)


def split_participants(
    data_paths: list[str | Path],
    train_percentage: int = 60,
):
    # Split all participants from datasets in data_paths into train, val and test splits
    participants = []
    for data_path in data_paths:
        with xr.open_dataset(data_path) as ds:
            participants.extend(ds.participant.values)
    # Ensure no duplication of participants
    participants = list(dict.fromkeys(participants))
    # Find amounts of train and test/val participants
    train_n = int(len(participants) * (train_percentage / 100))
    testval_n = len(participants) - train_n

    # Split into train, test, and val by sampling randomly
    testval_participants = random.sample(participants, testval_n)
    train_participants = [p for p in participants if p not in testval_participants]
    val_participants = testval_participants[: testval_n // 2]
    test_participants = testval_participants[testval_n // 2 :]
    return (train_participants, val_participants, test_participants)


def get_folds(
    data: xr.Dataset,
    k: int,
) -> list[np.ndarray]:
    """Divides dataset into folds

    Args:
        data (xr.Dataset): Dataset to be used
        k (int): Amount of folds

    Raises:
        ValueError: Occurs when k does not divide number of participants

    Returns:
        list[np.ndarray]: List of folds
    """
    # Make sure #participants is divisible by k
    n_participants = len(data.participant)
    if n_participants % k != 0:
        raise ValueError(
            f"K: {k} (amount of folds) must divide number of participants: {n_participants}"
        )

    # Divide data into k folds
    participants = data.participant.values.copy()
    np.random.shuffle(participants)
    folds = np.array_split(participants, k)
    return folds
