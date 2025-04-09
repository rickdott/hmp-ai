from torch.utils.data import Dataset
from torchvision.transforms import Compose
import netCDF4
import xarray as xr
import torch
import numpy as np
from hmpai.pytorch.normalization import *
from hmpai.pytorch.utilities import add_relative_positional_encoding
from hmpai.utilities import get_masking_index, MASKING_VALUE
from hmpai.data import SAT_CLASSES_ACCURACY
from pathlib import Path
from typing import Callable

global_ds_cache = {}


class MultiXArrayProbaDataset(Dataset):
    def __init__(
        self,
        data_paths: list[str | Path],
        participants_to_keep: list = None,
        labels: list[str] = SAT_CLASSES_ACCURACY,
        info_to_keep: list[str] = [],
        transform: Compose = None,
        normalization_fn: Callable[
            [torch.Tensor, float, float], torch.Tensor
        ] = norm_dummy,
        norm_vars: tuple[float, float] = None,
        subset_cond: tuple = None,
        statistics: dict = None,
        add_negative: bool = False,
        probabilistic_labels: bool = False,
        skip_samples: int = 0,
        cut_samples: int = 0,
        data_labels: list[
            list
        ] = None,
        add_pe: bool = False,
        subset_channels: list = None,
    ):
        """
        Initializes the data generator with the specified parameters.

        Args:
            data_paths (list[str | Path]): List of paths to the datasets.
            participants_to_keep (list, optional): List of participants to include. Defaults to None (uses all participants).
            labels (list[str], optional): List of labels to use. Defaults to SAT_CLASSES_ACCURACY.
            info_to_keep (list[str], optional): List of additional information to retain. Defaults to an empty list.
            transform (Compose, optional): Transformation to apply to the data. Defaults to None.
            normalization_fn (Callable[[torch.Tensor, float, float], torch.Tensor], optional): 
                Function to normalize the data. Defaults to norm_dummy.
            norm_vars (tuple[float, float], optional): Normalization variables (mean, std). Defaults to None.
            subset_cond (tuple, optional): Condition to subset the data, specified as 
                (variable, method, value). Examples: ('event_name', 'contains', 'speed') or 
                ('condition', 'equal', 'long'). Defaults to None.
            statistics (dict, optional): Precomputed statistics for normalization. Defaults to None.
            add_negative (bool, optional): Whether to add negative samples. Defaults to False.
            probabilistic_labels (bool, optional): Whether to return labels as probabilities over 
                each class instead of categorical. Defaults to False.
            skip_samples (int, optional): Number of samples to skip (e.g., pre-stimulus samples). Defaults to 0.
            cut_samples (int, optional): Number of samples to take from the end (e.g., post-response offset samples). Defaults to 0.
            data_labels (list[list], optional): List of labels for each dataset, used when datasets 
                have different labels. Defaults to None.
            add_pe (bool, optional): Whether to add positional encoding. Defaults to False.
            subset_channels (list, optional): List of channels to subset. Defaults to None.

        Attributes:
            data_paths (list[str | Path]): Paths to the datasets.
            data_labels (list[list]): Labels for each dataset.
            transform (Compose): Transformation applied to the data.
            info_to_keep (list[str]): Additional information retained.
            keep_info (bool): Whether additional information is retained.
            participants_to_keep (list): List of participants to include.
            max_length (int): Maximum length of the datasets.
            labels (list[str]): List of labels.
            label_lookup (dict): Mapping of labels to indices.
            subset_cond (tuple): Condition to subset the data.
            statistics (dict): Precomputed statistics for normalization.
            add_negative (bool): Whether negative samples are added.
            probabilistic_labels (bool): Whether labels are returned as probabilities.
            skip_samples (int): Number of samples skipped.
            cut_samples (int): Number of samples taken from the end.
            add_pe (bool): Whether positional encoding is added.
            subset_channels (list): List of channels to subset.
            index_map (dict): Mapping of indices for the dataset.
            normalization_fn (Callable): Function used for normalization.
            norm_vars (tuple[float, float]): Normalization variables (mean, std).
        """
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transform = transform
        self.info_to_keep = info_to_keep
        self.keep_info = len(self.info_to_keep) > 0
        self.participants_to_keep = (
            participants_to_keep if participants_to_keep is not None else []
        )

        # Find max length in datasets
        self.max_length = 0
        for data_path in data_paths:
            ds = xr.open_dataset(data_path)
            if len(ds.samples) > self.max_length:
                self.max_length = len(ds.samples)
            ds.close()

        self.labels = labels
        self.label_lookup = {label: idx for idx, label in enumerate(labels)}

        # Will return (n_events, samples) probability distribution as labels
        self.subset_cond = subset_cond
        self.statistics = statistics
        self.add_negative = add_negative
        self.probabilistic_labels = probabilistic_labels
        self.skip_samples = skip_samples
        self.cut_samples = cut_samples
        self.add_pe = add_pe
        self.subset_channels = subset_channels

        self.index_map = self._create_index_map_whole()

        if norm_vars is None:
            if self.statistics is None:
                self._calc_global_statistics(min(1000, self.__len__()))
            norm_vars = get_norm_vars_from_global_statistics(
                self.statistics, normalization_fn
            )

        self.normalization_fn = normalization_fn
        self.norm_vars = norm_vars

    def _create_index_map_whole(self):
        index_map = []

        for file_idx, file_path in enumerate(self.data_paths):
            with xr.open_dataset(file_path) as ds:
                # Find trials for which all samples for one channel are NaN, these are excluded
                data = ds["data"].values
                probas = ds["probabilities"].values
                # Select subset of data to check valid indices
                data = data[..., 0, :]
                mask = ~np.isnan(data).all(axis=-1) & ~(probas.sum(axis=-1) == 0).all(
                    axis=-1
                )
                if self.subset_cond is not None:
                    if self.subset_cond[1] == "contains":
                        subset_indices = ds[self.subset_cond[0]].str.contains(
                            self.subset_cond[2]
                        )
                    elif self.subset_cond[1] == "equal":
                        subset_indices = ds[self.subset_cond[0]] == self.subset_cond[2]
                    subset_indices = subset_indices.to_numpy()

                combined_indices = (
                    np.argwhere((mask) & (subset_indices))
                    if self.subset_cond is not None
                    else np.argwhere(mask)
                )

                if len(self.participants_to_keep) > 0:
                    participants_in_data = [
                        index
                        for index, value in enumerate(ds.participant.values.tolist())
                        if value in self.participants_to_keep
                    ]
                    index_map.extend(
                        [
                            (file_idx, *idx)
                            for idx in combined_indices
                            if idx[0] in participants_in_data
                        ]
                    )
                else:
                    index_map.extend([(file_idx, *idx) for idx in combined_indices])

        return index_map

    def _calc_global_statistics(self, sample_size):
        sample_size = (
            sample_size if len(self.index_map) >= sample_size else len(self.index_map)
        )
        # Normalization variables have not been calculated, compute these by sampling from the dataset
        indices = np.random.choice(range(len(self.index_map)), (sample_size,))

        global_min = float("inf")
        global_max = float("-inf")
        n_samples = 0
        global_sum = 0.0
        global_sum_squares = 0.0
        all_data = []
        label_counter = {k: 0 for k in self.label_lookup.values()}

        for idx in indices:
            data, label = self.__getitem_clean__(idx)
            data = data.numpy()

            sample_min = np.nanmin(data)
            sample_max = np.nanmax(data)
            global_min = min(global_min, sample_min)
            global_max = max(global_max, sample_max)

            valid_data = np.nan_to_num(data, nan=0.0)
            nan_mask = np.isnan(data)
            global_sum += np.sum(valid_data)
            global_sum_squares += np.sum(valid_data**2)
            n_samples += np.sum(~nan_mask)

            all_data.append(valid_data[~nan_mask])

        all_data = np.concatenate(all_data)

        global_mean = global_sum / n_samples
        global_std = np.sqrt(global_sum_squares / n_samples - global_mean**2)

        total_labels = sum(label_counter.values())
        class_weights = {
            label: 0 if count == 0 else total_labels / count
            for label, count in label_counter.items()
        }

        median = np.median(all_data)
        mad = np.median(np.abs(all_data - median))

        class_weights = torch.Tensor(list(class_weights.values()))
        self.statistics = {
            "global_min": global_min,
            "global_max": global_max,
            "global_mean": global_mean,
            "global_std": global_std,
            "global_median": median,
            "mad_score": mad,
            "class_weights": class_weights,
        }

    def _get_dataset(self, file_idx):
        file_path = self.data_paths[file_idx]
        if file_path not in global_ds_cache:
            global_ds_cache[file_path] = xr.open_dataset(file_path)
        return global_ds_cache[file_path]

    def __getitem__(self, idx, debug=False):
        indices = self.index_map[idx]

        ds = self._get_dataset(indices[0])
        pad_left = 0
        pad_right = 0
        filter = {
            "participant": indices[1],
            "epochs": indices[2],
        }
        sample = ds.isel(**filter)
        if len(sample.samples) < self.max_length:
            pad_right += self.max_length - len(sample.samples)

        # Subset channels
        if self.subset_channels is not None:
            sample = sample.sel(channels=self.subset_channels)
        sample_data = torch.as_tensor(sample.data.values, dtype=torch.float32)
        if pad_left > 0 or pad_right > 0:
            sample_data = torch.nn.functional.pad(
                sample_data, (pad_left, pad_right), mode="constant", value=torch.nan
            )

        sample_label = torch.as_tensor(sample.probabilities.values, dtype=torch.float32)
        if pad_left > 0 or pad_right > 0:
            sample_label = torch.nn.functional.pad(
                sample_label, (pad_left, pad_right), mode="constant", value=0
            )

        # Convert label probabilities to correct order
        if self.data_labels is not None:
            ds_labels = self.data_labels[indices[0]]
            ds_label_indices = [self.labels.index(label) for label in ds_labels]
            new_labels = torch.zeros(
                (len(self.labels), sample_label.shape[1]), dtype=torch.float32
            )
            for old_idx, new_idx in enumerate(ds_label_indices):
                new_labels[new_idx] = sample_label[old_idx]
            sample_label = new_labels

        if self.add_negative:
            sample_label[0, :] = 1 - sample_label.sum(axis=0)
        sample_label = sample_label.transpose(1, 0)

        # Swap samples and channels dims, since [time, features] is expected
        sample_data = sample_data.transpose(1, 0)

        end_idx = get_masking_index(sample_data, search_value=torch.nan)
        sample_data[end_idx - self.cut_samples : end_idx, :] = torch.nan

        sample_label[end_idx - self.cut_samples : end_idx, :] = 0
        if self.add_negative:
            sample_label[end_idx - self.cut_samples : end_idx, 0] = 1.0
        sample_label = sample_label[self.skip_samples :, :]

        sample_data = sample_data[self.skip_samples :, :]

        if self.transform is not None:
            sample_data, sample_label = self.transform((sample_data, sample_label))

        sample_data = self.normalization_fn(sample_data, *self.norm_vars)

        # fillna with masking_value
        sample_data = torch.nan_to_num(sample_data, nan=MASKING_VALUE)

        # Add positional encoding
        if self.add_pe:
            sample_data, sample_label = add_relative_positional_encoding(
                (sample_data, sample_label)
            )

        if self.keep_info:
            values_to_keep = [
                np.atleast_1d(sample[key].to_numpy()) for key in self.info_to_keep
            ]
            sample_info = [
                {key: value for key, value in zip(self.info_to_keep, values)}
                for values in zip(*values_to_keep)
            ]
            return sample_data, sample_label, sample_info
        return sample_data, sample_label

    def __getitem_clean__(self, idx):
        # For use in calculating normalization variables and class weights
        indices = self.index_map[idx]
        ds = self._get_dataset(indices[0])
        n_samples = len(ds.samples)

        pad_left = 0
        pad_right = 0
        filter = {
            "participant": indices[1],
            "epochs": indices[2],
        }
        sample = ds.isel(**filter)
        sample_data = torch.as_tensor(sample.data.values, dtype=torch.float32)
        if pad_left > 0 or pad_right > 0:
            sample_data = torch.nn.functional.pad(
                sample_data,
                (pad_left, pad_right),
                mode="constant",
                value=torch.nan,
            )

        sample_label = torch.as_tensor(
            sample.probabilities.values, dtype=torch.float32
        ).transpose(1, 0)

        return sample_data, sample_label

    def __len__(self):
        return len(self.index_map)
