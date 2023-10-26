import xarray as xr


def norm_0_to_1(dataset: xr.Dataset, min: float, max: float) -> xr.Dataset:
    dataset['data'] = (dataset['data'] - min) / (max - min)
    return dataset


def norm_min1_to_1(dataset: xr.Dataset, min: float, max: float) -> xr.Dataset:
    dataset['data'] = 2 * (dataset['data'] - min) / (max - min) - 1
    return dataset


def norm_dummy(dataset: xr.Dataset, min: float, max: float) -> xr.Dataset:
    return dataset
