import random
import xarray as xr
from typing import Callable
from hmpai.normalization import norm_0_to_1


def split_data_on_participants(
    data: xr.Dataset,
    train_percentage: int = 60,
    normalization_fn: Callable[[xr.Dataset, float, float], xr.Dataset] = norm_0_to_1,
) -> (xr.Dataset, xr.Dataset, xr.Dataset):
    """Splits dataset into three distinct sets based on participant, ensuring
    that no participant occurs in more than one set.
    Splits remainder of train percentage into two sets.
    Also normalizes data based on training set parameters to prevent information leakage.

    Args:
        data (xr.Dataset): Dataset to be split.
        train_percentage (int): Percentage of participants used in the training set. Defaults to 60.
        normalization_fn (Callable[[xr.Dataset, float, float], xr.Dataset], optional): Normalization function to use. Defaults to norm_0_to_1.

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
    train_participants = [p for p in participants if p not in testval_participants]
    val_participants = testval_participants[: testval_n // 2]
    test_participants = testval_participants[testval_n // 2 :]

    # Select subsets from data
    train_data = data.sel(participant=train_participants)
    val_data = data.sel(participant=val_participants)
    test_data = data.sel(participant=test_participants)

    # Normalize data
    train_min = train_data.min(skipna=True).data.item()
    train_max = train_data.max(skipna=True).data.item()

    train_data = normalization_fn(train_data, train_min, train_max)
    val_data = normalization_fn(val_data, train_min, train_max)
    test_data = normalization_fn(test_data, train_min, train_max)

    return train_data, val_data, test_data
