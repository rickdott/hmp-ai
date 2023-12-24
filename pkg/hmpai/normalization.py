import xarray as xr


def norm_0_to_1(dataset: xr.Dataset, min: float, max: float) -> xr.Dataset:
    dataset['data'] = (dataset['data'] - min) / (max - min)
    return dataset


def norm_min1_to_1(dataset: xr.Dataset, min: float, max: float) -> xr.Dataset:
    dataset['data'] = 2 * (dataset['data'] - min) / (max - min) - 1
    return dataset


def norm_zscore(dataset: xr.Dataset, mean: float, std: float) -> xr.Dataset:
    dataset['data'] = (dataset['data'] - mean) / std
    return dataset


def norm_dummy(dataset: xr.Dataset, min: float, max: float) -> xr.Dataset:
    return dataset


def get_norm_vars(dataset: xr.Dataset, norm_fn) -> (float, float):
    """
    Calculate the normalization variables for a given dataset. Mean/std if Z-score, otherwise min/max

    Args:
        dataset (xr.Dataset): The dataset to calculate the normalization variables for.
        norm_fn: The normalization function to use.

    Returns:
        A tuple containing the calculated normalization variables.
    """
    if norm_fn == norm_zscore:
        return dataset['data'].mean(skipna=True), dataset['data'].std(skipna=True)
    else:
        return dataset['data'].min(skipna=True), dataset['data'].max(skipna=True)