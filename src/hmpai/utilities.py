import numpy as np
import json
import math
import pandas as pd
import torch
import xarray as xr
import seaborn as sns
import matplotlib

# Value that means data should not be used in training
MASKING_VALUE = 999


def pretty_json(data: dict) -> str:
    # From https://www.tensorflow.org/tensorboard/text_summaries
    json_data = json.dumps(data, indent=2)
    return "".join(f"\t{line}" for line in json_data.splitlines(True))


def get_masking_indices(t, search_value=MASKING_VALUE):
    # Expects a batch as input: [batch_size, time, channels]
    # Also use this one if epoch is unsqueezed
    if isinstance(search_value, float) and math.isnan(search_value):
        mask = torch.isnan(t[:, :, 0])
    elif torch.is_tensor(search_value) and torch.isnan(search_value):
        mask = torch.isnan(t[:, :, 0])
    else:
        mask = t[:, :, 0] == search_value
    reversed_mask = torch.flip(mask, dims=[1])
    last_block_start = (~reversed_mask).float().argmax(dim=1)
    max_indices = mask.shape[1] - last_block_start
    return max_indices


def get_masking_index(t, search_value=MASKING_VALUE):
    # Expects a single epoch as input: [time, channels]
    if isinstance(search_value, float) and math.isnan(search_value):
        mask = torch.isnan(t[:, 0])
    elif torch.is_tensor(search_value) and torch.isnan(search_value):
        mask = torch.isnan(t[:, 0])
    else:
        mask = t[:, 0] == search_value
    reversed_mask = torch.flip(mask, dims=[0])
    last_block_start = (~reversed_mask).float().argmax(dim=0)
    max_index = mask.shape[0] - last_block_start
    return max_index


def get_masking_indices_xr(data: xr.DataArray, search_value=MASKING_VALUE):
    # Check if search_value is NaN
    if isinstance(search_value, float) and np.isnan(search_value):
        mask = np.isnan(
            data.isel(channels=0)
        )  # Select the first channel and apply NaN mask
    else:
        mask = (
            data.isel(channels=0) == search_value
        )  # Comparison for non-NaN search values

    # Reverse mask along the time dimension
    reversed_mask = mask.isel(samples=slice(None, None, -1))

    # Find the first occurrence of non-mask values in the reversed mask
    last_block_start = (~reversed_mask).argmax(dim="samples")

    # Calculate the max indices based on the mask shape and block start positions
    max_indices = (
        mask.shape[1] - last_block_start.values
    )  # Adjusting based on time dimension length

    return max_indices


def get_trial_start_end(
    probabilities: torch.Tensor, lbl_start: int = 1, lbl_len: int = 3
):
    # Create a mask where any non-zero value exists along the channels
    # Any non-negative class
    mask = (probabilities[:, lbl_start : lbl_start + lbl_len] != 0).any(dim=1)

    # Find the first and last non-zero indices
    if mask.any():
        nonzero_mask = torch.nonzero(mask, as_tuple=False)
        first_nonzero = nonzero_mask[0].item()
        last_nonzero = nonzero_mask[-1].item()
    else:
        first_nonzero, last_nonzero = -1, -1  # Default if all values are zero

    lowest_highest = (first_nonzero, last_nonzero)

    return lowest_highest


def set_seaborn_style():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["font.size"] = 7
    sns.set_style("ticks")
    sns.set_context("paper")
    # sns.set_palette(
    #     sns.color_palette(
    #         [
    #             "#4477AA",
    #             "#66CCEE",
    #             "#228833",
    #             "#CCBB44",
    #             "#EE6677",
    #             "#AA3377",
    #         ]
    #     )
    # )
    sns.set_palette(
        sns.color_palette(
            palette=[
                # existing six
                "#4477aa",
                "#66ccee",
                "#228833",
                "#ccbb44",
                "#ee6677",
                "#aa3377",
                # extension
                "#44aa99",
                "#aa7744",
                "#ddaa33",
                "#999933",
                "#bb5566",
                "#7777bb",
            ]
        )
    )


def calc_ratio(
    data: pd.DataFrame, column: str, rt_col: str = "rt_x", normalize: bool = True
):
    auc_column = column + "_auc"
    ratio_column = column + "_ratio"
    # Assuming rt_col is in seconds
    data[ratio_column] = data[auc_column] / (data[rt_col] * 250)

    # z-score
    if normalize:
        data[ratio_column] = (data[ratio_column] - data[ratio_column].mean()) / data[
            ratio_column
        ].std()

    return data


def get_p(p):
    if p < 0.001:
        return "< 0.001"
    if p < 0.01:
        return "< 0.01"
    if p < 0.05:
        return "< 0.05"
    return f"= {p:.2f}"


def format_stats_latex(model):
    for index, row in model.coefs.iterrows():
        print(index)
        print(
            f"($\\beta = {row['Estimate']:.2f}$, $SE = {row['SE']:.2f}$, $z = {row['Z-stat']:.2f}$, $p {get_p(row['P-val'])}$, $OR = {row['OR']:.2f}$, $95\\%\\,CI\\,[{row['OR_2.5_ci']:.2f}, {row['OR_97.5_ci']:.2f}]$)"
        )
