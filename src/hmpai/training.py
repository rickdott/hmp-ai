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
    data_paths: list[str | Path], n_folds: int, shuffle: bool = True
):
    participants = []
    for data_path in data_paths:
        with xr.open_dataset(data_path) as ds:
            participants.extend(ds.participant.values)

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
    # Find amounts of train and test/val participants
    train_n = int(len(participants) * (train_percentage / 100))
    testval_n = len(participants) - train_n

    # Split into train, test, and val by sampling randomly
    testval_participants = random.sample(participants, testval_n)
    train_participants = [p for p in participants if p not in testval_participants]
    val_participants = testval_participants[: testval_n // 2]
    test_participants = testval_participants[testval_n // 2 :]
    return (train_participants, val_participants, test_participants)


def split_data_on_participants(
    data: xr.Dataset,
    train_percentage: int = 60,
    normalization_fn: Callable[[xr.Dataset, float, float], xr.Dataset] = norm_dummy,
    truncate_sample: int = None,
) -> (xr.Dataset, xr.Dataset, xr.Dataset):
    """Splits dataset into three distinct sets based on participant, ensuring
    that no participant occurs in more than one set.
    Splits remainder of train percentage into two sets.
    Also normalizes data based on training set parameters to prevent information leakage.

    Args:
        data (xr.Dataset): Dataset to be split.
        train_percentage (int): Percentage of participants used in the training set. Defaults to 60.
        normalization_fn (Callable[[xr.Dataset, float, float], xr.Dataset], optional): Normalization function to use. Defaults to norm_0_to_1.
        truncate_sample (int, optional): Number of samples to truncate to. Defaults to None.

    Returns:
        (xr.Dataset, xr.Dataset, xr.Dataset): tuple of train, test, val datasets.
    """
    participants = data.participant.values.tolist()
    # In case of SAT1 experiment, 25 participants are used
    # Given train_percentage=60, remaining 40 percent will be split evenly between validation and test sets
    # 100-train_percentage must be divisible by 2

    # Find amounts of train and test/val participants
    train_n = int(len(participants) * (train_percentage / 100))
    testval_n = len(participants) - train_n

    # Split into train, test, and val by sampling randomly
    testval_participants = random.sample(participants, testval_n)
    print(testval_participants)
    train_participants = [p for p in participants if p not in testval_participants]
    val_participants = testval_participants[: testval_n // 2]
    test_participants = testval_participants[testval_n // 2 :]

    # Select subsets from data
    if truncate_sample is not None:
        train_data = data.sel(
            participant=train_participants, samples=slice(0, truncate_sample - 1)
        )
        val_data = data.sel(
            participant=val_participants, samples=slice(0, truncate_sample - 1)
        )
        test_data = data.sel(
            participant=test_participants, samples=slice(0, truncate_sample - 1)
        )
    else:
        train_data = data.sel(participant=train_participants)
        val_data = data.sel(participant=val_participants)
        test_data = data.sel(participant=test_participants)

    # Normalize data
    norm_var1, norm_var2 = get_norm_vars(train_data, normalization_fn)

    train_data = normalization_fn(train_data, norm_var1, norm_var2)
    val_data = normalization_fn(val_data, norm_var1, norm_var2)
    test_data = normalization_fn(test_data, norm_var1, norm_var2)

    return train_data, val_data, test_data


def split_index_map_tueg(
    index_map: pd.DataFrame, train_percentage: int = 60, include_test: bool = False
):
    participants = index_map["participant"].unique()

    train_n = int(len(participants) * (train_percentage / 100))
    testval_n = len(participants) - train_n

    testval_participants = np.random.choice(participants, testval_n)
    train_participants = participants[~np.isin(participants, testval_participants)]

    if include_test:
        val_participants = testval_participants[: testval_n // 2]
        test_participants = testval_participants[testval_n // 2 :]
        return (
            index_map[index_map["participant"].isin(train_participants)],
            index_map[index_map["participant"].isin(val_participants)],
            index_map[index_map["participant"].isin(test_participants)],
        )

    return (
        index_map[index_map["participant"].isin(train_participants)],
        index_map[index_map["participant"].isin(testval_participants)],
    )


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


def k_fold_cross_validate_sklearn(
    model: sklearn.base.ClassifierMixin,
    data: xr.Dataset,
    k: int = 5,
    normalization_fn: Callable[[xr.Dataset, float, float], xr.Dataset] = norm_dummy,
    seed: int = 42,
):
    results = defaultdict(list)
    set_global_seed(seed)
    folds = get_folds(data, k)

    for i_fold in range(len(folds)):
        train_folds = deepcopy(folds)
        test_fold = train_folds.pop(i_fold)
        train_fold = np.concatenate(train_folds, axis=0)
        print(f"Fold {i_fold + 1}: test fold: {test_fold}")

        train_data = data.sel(participant=train_fold)
        test_data = data.sel(participant=test_fold)

        # Normalize data
        norm_var1, norm_var2 = get_norm_vars(train_data, normalization_fn)
        train_data = normalization_fn(train_data, norm_var1, norm_var2)
        test_data = normalization_fn(test_data, norm_var1, norm_var2)

        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

        train_data = calculate_features(train_data)
        test_data = calculate_features(test_data)

        train_data_np = train_data.to_numpy().reshape(
            -1, train_data.shape[1] * train_data.shape[2]
        )
        test_data_np = test_data.to_numpy().reshape(
            -1, test_data.shape[1] * test_data.shape[2]
        )

        clf = model.fit(train_data_np, train_data.labels)

        predictions = clf.predict(test_data_np)

        result = classification_report(test_data.labels, predictions, output_dict=True)
        print(f"Fold {i_fold + 1}: Accuracy: {result['accuracy']}")
        print(f"Fold {i_fold + 1}: F1-Score: {result['macro avg']['f1-score']}")
        # Does not support multiple test sets
        results[0].append(result)
    return results


def calculate_features(data: xr.Dataset) -> xr.Dataset:
    """Calculates features from data

    Args:
        data (xr.Dataset): Dataset to calculate features from

    Returns:
        xr.Dataset: Dataset with features
    """
    data = data.where(data.data != MASKING_VALUE)
    new_dataset = xr.Dataset()

    # Calculate features
    new_dataset["mean"] = data["data"].mean(dim="samples")
    new_dataset["std"] = data["data"].std(dim="samples")
    new_dataset["min"] = data["data"].min(dim="samples")
    new_dataset["max"] = data["data"].max(dim="samples")
    new_dataset["median"] = data["data"].median(dim="samples")
    new_dataset["var"] = data["data"].var(dim="samples")

    new_dataset = xr.concat(
        [new_dataset[var] for var in new_dataset.data_vars], dim="features"
    )
    new_dataset = new_dataset.transpose("index", "channels", "features")

    return new_dataset
