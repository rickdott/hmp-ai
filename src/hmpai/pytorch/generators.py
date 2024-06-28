from torch.utils.data import Dataset
from torchvision.transforms import Compose
import netCDF4
import xarray as xr
import torch
import numpy as np
from hmpai.data import SAT1_STAGES_ACCURACY, preprocess, MASKING_VALUE
from hmpai.pytorch.normalization import *
from hmpai.pytorch.utilities import DEVICE
import pyedflib
from pathlib import Path
from functools import lru_cache
from typing import Callable


class SAT1Dataset(Dataset):
    """
    PyTorch dataset for the SAT-1 dataset.

    Args:
        dataset (xr.Dataset): The input dataset.
        info_to_keep (list[str]): list of XArray dimension names to keep in a dictionary, returned as the third index of the batch item.
        shape_topological (bool): Whether to shape the data topologically.
        do_preprocessing (bool): Whether to preprocess the data.

    Attributes:
        data (torch.Tensor): The preprocessed data.
        labels (torch.Tensor): The labels for the data.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        shape_topological=False,
        do_preprocessing=True,
        labels: list[str] = SAT1_STAGES_ACCURACY,
        set_to_zero: bool = False,
        info_to_keep: list[str] = [],
        interpolate_to: int = 0,
        order_by_rt: bool = False,
        transform: Compose = None,
    ):
        self.interpolate_to = interpolate_to
        self.transform = transform
        # Alphabetical ordering of labels used for categorization of labels
        label_lookup = {label: idx for idx, label in enumerate(labels)}

        # If labels is a data variable, the data is sequential instead of split
        sequential = (
            dataset.data_vars.__contains__("labels") and "probabilities" not in dataset
        )

        # Preprocess data
        if do_preprocessing:
            dataset = preprocess(
                dataset,
                shuffle=True,
                shape_topological=shape_topological,
                sequential=sequential,
            )

        values_to_keep = [dataset[key].to_numpy() for key in info_to_keep]
        self.info = [dict(zip(info_to_keep, values)) for values in zip(*values_to_keep)]

        self.data = torch.as_tensor(dataset.data.to_numpy(), dtype=torch.float32)

        if "probabilities" in dataset:
            self.labels = torch.as_tensor(
                dataset.probabilities.values, dtype=torch.float32
            )
        else:
            vectorized_label_to_index = np.vectorize(lambda x: label_lookup.get(x, -1))
            indices = xr.apply_ufunc(vectorized_label_to_index, dataset.labels)
            self.labels = torch.as_tensor(indices.values, dtype=torch.long)
            if set_to_zero:
                self.labels = torch.where(
                    self.labels == -1, torch.tensor(0), self.labels
                )
        if interpolate_to != 0:
            self.data = torch.Tensor(self.__resample_batch_eeg__(self.data))
            # Probably does not work with probabilities
            self.labels = torch.Tensor(
                self.__resample_batch_labels__(self.labels)
            ).long()
        if order_by_rt:
            if "rt" not in info_to_keep:
                raise ValueError(
                    "rt must be included in info_to_keep and in the source data to be able to order by rt."
                )
            combined = list(zip(self.data, self.labels, self.info))
            sorted_combined = sorted(combined, key=lambda x: x[2]["rt"])
            self.data, self.labels, self.info = zip(*sorted_combined)
            self.data = torch.stack(self.data)
            self.labels = torch.stack(self.labels).long()
            self.info = list(self.info)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            data, labels = self.transform((self.data[idx], self.labels[idx]))
            # data = self.transform(self.data[idx])
            # labels = self.labels[idx]
        else:
            data = self.data[idx]
            labels = self.labels[idx]
        if len(self.info) > 0:
            return data, labels, self.info[idx]
        else:
            return data, labels

    def __resample_batch_eeg__(self, trials):
        batch_size, original_length, num_channels = trials.shape

        # Create the original and target time axes
        original_time = np.linspace(0, 1, original_length)
        target_time = np.linspace(0, 1, self.interpolate_to)

        # Prepare an array to hold the resampled data
        resampled_trials = np.zeros((batch_size, self.interpolate_to, num_channels))

        # Perform the interpolation for each trial and each channel
        for i in range(batch_size):
            for j in range(num_channels):
                resampled_trials[i, :, j] = np.interp(
                    target_time, original_time, trials[i, :, j]
                )

        return resampled_trials

    def __resample_batch_labels__(self, labels):
        batch_size, original_length = labels.shape

        # Create the original and target time axes
        original_time = np.linspace(0, 1, original_length)
        target_time = np.linspace(0, 1, self.interpolate_to)

        # Prepare an array to hold the resampled labels
        resampled_labels = np.zeros(
            (batch_size, self.interpolate_to),
        )

        # Vectorized interpolation
        resampled_labels = np.array(
            [
                np.interp(target_time, original_time, labels[i])
                for i in range(batch_size)
            ],
            dtype=np.intc,
        )

        return resampled_labels


global_ds_cache = {}


class MultiXArrayDataset(Dataset):
    def __init__(
        self,
        data_paths: list[str | Path],
        participants_to_keep: list = None,
        do_preprocessing: bool = True,
        labels: list[str] = SAT1_STAGES_ACCURACY,
        set_to_zero: bool = False,
        info_to_keep: list[str] = [],
        transform: Compose = None,
        normalization_fn: Callable[
            [torch.Tensor, float, float], torch.Tensor
        ] = norm_dummy,
        norm_vars: tuple[float, float] = None,
    ):
        self.data_paths = data_paths
        self.transform = transform
        self.info_to_keep = info_to_keep
        self.keep_info = len(self.info_to_keep) > 0
        self.participants_to_keep = (
            participants_to_keep if participants_to_keep is not None else []
        )

        self.label_lookup = {label: idx for idx, label in enumerate(labels)}
        self.index_map = self._create_index_map()
        
        # Open first dataset to check if split
        with xr.open_dataset(data_paths[0]) as ds:
            self.split = "labels" not in ds.data_vars and "probabilities" not in ds

        if norm_vars is None:
            self._calc_global_statistics(4000)
            norm_vars = (self.statistics['global_min'], self.statistics['global_max']) if normalization_fn != norm_zscore else (self.statistics['global_mean'], self.statistics['global_std'])

        self.normalization_fn = normalization_fn
        self.norm_vars = norm_vars


    def _create_index_map(self):
        index_map = []
        for file_idx, file_path in enumerate(self.data_paths):
            with xr.open_dataset(file_path) as ds:
                # num_participants * num_trials
                # Find trials for which all samples for one channel are NaN, these are excluded
                data = ds["data"].values
                # Select subset of data to check valid indices
                data = data[..., 0, :]
                mask = np.isnan(data).all(axis=-1)
                indices = np.argwhere(~mask)
                # Add indices to index_map, filtering based on required participants
                if len(self.participants_to_keep) > 0:
                    participants_in_data = [
                        index
                        for index, value in enumerate(ds.participant.values.tolist())
                        if value in self.participants_to_keep
                    ]
                    index_map.extend([(file_idx, *idx) for idx in indices if idx[0] in participants_in_data])
                else:
                    index_map.extend([(file_idx, *idx) for idx in indices])
        return index_map

    def _get_dataset(self, file_idx):
        file_path = self.data_paths[file_idx]
        if file_path not in global_ds_cache:
            global_ds_cache[file_path] = xr.open_dataset(file_path)
        return global_ds_cache[file_path]

    def _calc_global_statistics(self, sample_size):
        sample_size = sample_size if len(self.index_map) >= sample_size else len(self.index_map)
        # Normalization variables have not been calculated, compute these by sampling from the dataset
        indices = np.random.choice(range(len(self.index_map)), (sample_size,))

        global_min = float('inf')
        global_max = float('-inf')
        n_samples = 0
        global_sum = 0.0
        global_sum_squares = 0.0
        label_counter = {k: 0 for k in self.label_lookup.values()}

        for idx in indices:
            data, label = self.__getitem_clean__(idx)
            data = data.numpy()
            label_counter[label] += 1

            sample_min = np.nanmin(data)
            sample_max = np.nanmax(data)
            global_min = min(global_min, sample_min)
            global_max = max(global_max, sample_max)

            valid_data = np.nan_to_num(data, nan=0.0)
            nan_mask = np.isnan(data)
            global_sum += np.sum(valid_data)
            global_sum_squares += np.sum(valid_data ** 2)
            n_samples += np.sum(nan_mask)

        global_mean = global_sum / n_samples
        global_std = np.sqrt(global_sum_squares / n_samples - global_mean ** 2)

        total_labels = sum(label_counter.values())
        class_weights = {label: total_labels / count for label, count in label_counter.items()}
        class_weights = torch.Tensor(list(class_weights.values()))
        self.statistics = {
            'global_min': global_min,
            'global_max': global_max,
            'global_mean': global_mean,
            'global_std': global_std,
            'class_weights': class_weights,
        }

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        indices = self.index_map[idx]

        ds = self._get_dataset(indices[0])
        filter = {"participant": indices[1], "epochs": indices[2]}
        if self.split:
            filter["labels"] = indices[3]

        sample = ds.isel(**filter)
        sample_data = torch.as_tensor(sample.data.values, dtype=torch.float32)
        sample_data = self.normalization_fn(sample_data, *self.norm_vars)

        # fillna with masking_value
        sample_data = torch.nan_to_num(sample_data, nan=MASKING_VALUE)
        # Swap samples and channels dims, since [time, features] is expected
        sample_data = sample_data.transpose(1, 0)
        # TODO: look into what to do when not using split
        sample_label = self.label_lookup[sample.labels.item()]

        if self.transform is not None:
            sample_data, sample_label = self.transform((sample_data, sample_label))
        if self.keep_info:
            values_to_keep = [sample[key].to_numpy() for key in self.info_to_keep]
            sample_info = [
                dict(zip(self.info_to_keep, values)) for values in zip(*values_to_keep)
            ]
            return sample_data, sample_label, sample_info
        return sample_data, sample_label
    
    def __getitem_clean__(self, idx):
        # For use in calculating normalization variables and class weights
        indices = self.index_map[idx]

        ds = self._get_dataset(indices[0])
        filter = {"participant": indices[1], "epochs": indices[2]}
        if self.split:
            filter["labels"] = indices[3]
        sample = ds.isel(**filter)
        sample_data = torch.as_tensor(sample.data.values, dtype=torch.float32)
        sample_label = self.label_lookup[sample.labels.item()]

        return sample_data, sample_label

    def __del__(self):
        # Ensure all datasets are closed when the object is deleted
        for file_path in self.data_paths:
            if file_path in global_ds_cache:
                global_ds_cache[file_path].close()
                del global_ds_cache[file_path]
        


class TUEGDataset(Dataset):
    def __init__(self, files_path):
        self.files = open(files_path, "r").readlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # One entry in self.files == one index, maybe split it up into individual samples?
        f = pyedflib.EdfReader(self.files[idx])
        return f
