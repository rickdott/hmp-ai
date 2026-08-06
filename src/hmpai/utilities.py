import mne
import numpy as np
import json
import math
import pandas as pd
import torch
import xarray as xr
import seaborn as sns
import matplotlib

import hmp

# Value that means data should not be used in training
MASKING_VALUE = 999


def pretty_json(data: dict) -> str:
    # From https://www.tensorflow.org/tensorboard/text_summaries
    json_data = json.dumps(data, indent=2)
    return "".join(f"\t{line}" for line in json_data.splitlines(True))


def get_masking_indices(t, search_value=MASKING_VALUE):
    # Expects a batch as input: [batch_size, time, channels]
    # Exclude the last channel from masking  (excludes PE if present, otherwise doesnt matter)
    t_excl_last = t[..., :-1]

    if isinstance(search_value, float) and math.isnan(search_value):
        mask = torch.isnan(t_excl_last).all(dim=-1)
    elif torch.is_tensor(search_value) and torch.isnan(search_value):
        mask = torch.isnan(t_excl_last).all(dim=-1)
    else:
        mask = (t_excl_last == search_value).all(dim=-1)

    # argmax finds the first True; falls back to 0 if none found
    first_mask_index = mask.float().argmax(dim=1)

    # If no masked timestep exists, return the full sequence length instead of 0
    has_mask = mask.any(dim=1)
    first_mask_index = torch.where(has_mask, first_mask_index, torch.tensor(mask.shape[1], device=t.device))

    return first_mask_index


def get_masking_index(t, search_value=MASKING_VALUE):
    # Expects a single epoch as input: [time, channels]
    # Exclude the last channel from masking check
    t_excl_last = t[:, :-1]

    if isinstance(search_value, float) and math.isnan(search_value):
        mask = torch.isnan(t_excl_last).all(dim=-1)
    elif torch.is_tensor(search_value) and torch.isnan(search_value):
        mask = torch.isnan(t_excl_last).all(dim=-1)
    else:
        mask = (t_excl_last == search_value).all(dim=-1)

    first_mask_index = mask.float().argmax(dim=0)

    # If no masked timestep exists, return the full sequence length instead of 0
    has_mask = mask.any()
    first_mask_index = first_mask_index if has_mask else torch.tensor(mask.shape[0], device=t.device)

    return first_mask_index


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

    # sns.set_palette(
    #     sns.color_palette(
    #         palette=[
    #             # existing six
    #             "#4477aa",
    #             "#66ccee",
    #             "#228833",
    #             "#ccbb44",
    #             "#ee6677",
    #             "#aa3377",
    #             # extension
    #             "#44aa99",
    #             "#aa7744",
    #             "#ddaa33",
    #             "#999933",
    #             "#bb5566",
    #             "#7777bb",
    #         ]
    #     )
    # )
    # sns.set_palette(sns.color_palette(
    #     palette=[
    #         (235/255, 172/255, 35/255),
    #         (184/255, 0/255, 88/255),
    #         (0/255, 140/255, 249/255),
    #         (0/255, 110/255, 0/255),
    #         (0/255, 187/255, 173/255),
    #         (209/255, 99/255, 230/255),
    #         (178/255, 69/255, 2/255),
    #         (255/255, 146/255, 135/255),
    #         (89/255, 84/255, 214/255),
    #         (0/255, 198/255, 248/255),
    #         (135/255, 133/255, 0/255),
    #         (0/255, 167/255, 108/255),
    #         (189/255, 189/255, 189/255),
    #     ]
    # ))
    
    sns.set_palette(
        sns.color_palette(
            palette=[
                "#E69F00",  # Orange
                "#56B4E9",  # Sky Blue
                "#009E73",  # Bluish Green
                "#CC79A7",  # Reddish Purple
                "#0072B2",  # Blue
                "#D55E00",  # Vermillion
                "#F0E442",  # Yellow

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


def adjust_offset(epoch_data: xr.Dataset, hmp_offset: float) -> xr.Dataset:
    sfreq = epoch_data.sfreq
    if 'offset' in epoch_data.attrs:
        offset_name = 'offset'
    else:
        offset_name = 'offset_end'
    tmp_offset = epoch_data.attrs[offset_name]
    hmp_offset = int(np.rint(hmp_offset * sfreq)) # 0.05 = 50 ms worth of samples for HMP after response, remainder is not used in HMP
    epoch_data = epoch_data.assign_attrs({offset_name: hmp_offset, f'extra_{offset_name}': tmp_offset - hmp_offset})

    return epoch_data


def add_splits_to_dataset(ds: xr.Dataset, splits: np.ndarray) -> xr.Dataset:
    split = xr.DataArray(np.full(ds.sizes["recording"], "unknown", dtype=object),
                         dims=["recording"],
                         coords={"recording": ds["recording"]})
    split.loc[splits[0]] = "train"
    split.loc[splits[1]] = "val"
    split.loc[splits[2]] = "test"

    ds = ds.assign_coords(split=split)
    return ds


def get_splits_from_dataset(ds: xr.Dataset) -> list[np.ndarray]:
    train_split = ds.recording.where(ds.split == 'train', drop=True).values
    val_split = ds.recording.where(ds.split == 'val', drop=True).values
    test_split = ds.recording.where(ds.split == 'test', drop=True).values

    return [train_split, val_split, test_split]


def read_mne_epochs(pfiles, montage='biosemi64', preprocessing_kwargs=None, subj_name=None, cpus=1):
    """Read MNE epochs from pfiles and return as xarray Dataset."""
    if preprocessing_kwargs is None:
        preprocessing_kwargs = {}

    # Read MNE epochs
    epoch_data, info = hmp.io.read_mne_epochs(pfiles, montage=montage, preprocessing_kwargs=preprocessing_kwargs, subj_name=subj_name, cpus=cpus)

    epoch_data.attrs['mne_info'] = info

    return epoch_data


def save_hmp_epochs(epoch_data: xr.Dataset, path: str):
    if 'mne_info' in epoch_data.attrs:
        info = epoch_data.attrs['mne_info']
        if isinstance(info, mne.Info):
            info_dict = info.to_json_dict()
            epoch_data.attrs['mne_info'] = json.dumps(info_dict)
    epoch_data.to_netcdf(path, engine="netcdf4")


def load_hmp_epochs(path):
    epoch_data = xr.load_dataset(path)
    if 'mne_info' in epoch_data.attrs:
        info_dict = json.loads(epoch_data.attrs['mne_info'])
        info = mne.Info.from_json_dict(info_dict)
        epoch_data.attrs['mne_info'] = info

    return epoch_data