import torch
import netCDF4
import xarray as xr
import numpy as np

def norm_0_to_1(tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return (tensor - min_val) / (max_val - min_val)

def norm_min1_to_1(tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return 2 * (tensor - min_val) / (max_val - min_val) - 1

def norm_zscore(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (tensor - mean) / std

def norm_dummy(tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return tensor

def get_norm_vars(tensor: torch.Tensor, norm_fn) -> tuple[float, float]:
    """
    Calculate the normalization variables for a given tensor. Mean/std if Z-score, otherwise min/max.

    Args:
        tensor (torch.Tensor): The tensor to calculate the normalization variables for.
        norm_fn: The normalization function to use.

    Returns:
        A tuple containing the calculated normalization variables.
    """
    if norm_fn == norm_zscore:
        return tensor.mean().item(), tensor.std().item()
    else:
        return tensor.min().item(), tensor.max().item()
    
def compute_global_statistics(data_paths, participants):
    global_min = float('inf')
    global_max = float('-inf')
    n_samples = 0
    global_sum = 0.0
    global_sum_squares = 0.0

    for file_path in data_paths:
        with xr.open_dataset(file_path) as ds:
            participants_in_data = [index for index, value in enumerate(ds.participant.values.tolist()) if value in participants]
            ds = ds.isel(participant=participants_in_data)
            data = ds['data'].values.astype(np.float32)
            
            # Update global min and max ignoring NaNs
            file_min = np.nanmin(data)
            file_max = np.nanmax(data)
            global_min = min(global_min, file_min)
            global_max = max(global_max, file_max)
            
            # Calculate sum and sum of squares ignoring NaNs
            valid_data = np.nan_to_num(data, nan=0.0)
            nan_mask = ~np.isnan(data)
            global_sum += np.sum(valid_data)
            global_sum_squares += np.sum(valid_data ** 2)
            n_samples += np.sum(nan_mask)
    
    global_mean = global_sum / n_samples
    global_std = np.sqrt(global_sum_squares / n_samples - global_mean ** 2)

    return global_min, global_max, global_mean, global_std