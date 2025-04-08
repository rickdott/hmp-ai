import random
from hmpai.visualization import plot_epoch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import netCDF4
import xarray as xr
import torch
import numpy as np
from hmpai.data import SAT1_STAGES_ACCURACY, preprocess, MASKING_VALUE
from hmpai.pytorch.normalization import *
from hmpai.pytorch.utilities import DEVICE, add_relative_positional_encoding
from hmpai.pytorch.transforms import ConcatenateTransform
from hmpai.utilities import get_masking_index, get_masking_indices
import pyedflib
from pathlib import Path
from functools import lru_cache
from typing import Callable
from tqdm.notebook import tqdm
import multiprocessing
from multiprocessing import Pool
import pandas as pd
import itertools
import h5py

global_ds_cache = {}

class MultiXArrayProbaDataset(Dataset):
    def __init__(
        self,
        data_paths: list[str | Path],
        participants_to_keep: list = None,
        labels: list[str] = SAT1_STAGES_ACCURACY,
        info_to_keep: list[str] = [],
        transform: Compose = None,
        normalization_fn: Callable[
            [torch.Tensor, float, float], torch.Tensor
        ] = norm_dummy,
        norm_vars: tuple[float, float] = None,
        window_size: tuple[int, int] = (5, 5),
        jiggle: int = 3,
        whole_epoch: bool = False,
        subset_cond: tuple = None,  # (variable, method, value), examples: ('event_name', 'contains', 'speed') or ('condition', 'equal', 'long')
        statistics: dict = None,
        add_negative: bool = False,
        probabilistic_labels: bool = False,  # Return labels as probabilities over each class instead of categorical
        skip_samples: int = 0,  # Skip this amount of samples, used to skip pre-stimulus samples
        cut_samples: int = 0,  # Take this amount of samples from the end, used to skip extra post-response offset samples
        data_labels: list[
            list
        ] = None,  # List of equal length as data_paths in case datasets have different labels, labels param should in this case be a list containing the intersection of all of these
        add_pe: bool = False,
        subset_channels: list = None,
    ):
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

        electrode_mapping = {
            "EEG FP1": "Fp1",
            "EEG FP2": "Fp2",
            "EEG F7": "F7",
            "EEG F8": "F8",
            "EEG F3": "F3",
            "EEG F4": "F4",
            "EEG FZ": "Fz",
            "EEG T3": "T7",  # T3 corresponds to T7
            "EEG T4": "T8",  # T4 corresponds to T8
            "EEG C3": "C3",
            "EEG C4": "C4",
            "EEG CZ": "Cz",
            "EEG T5": "P7",  # T5 corresponds to P7
            "EEG T6": "P8",  # T6 corresponds to P8
            "EEG P3": "P3",
            "EEG P4": "P4",
            "EEG PZ": "Pz",
            "EEG O1": "O1",
            "EEG O2": "O2",
        }
        self.sat2_chs = list(electrode_mapping.values())
        self.ch_names = [ch.upper() for ch in self.sat2_chs]

        self.labels = labels
        self.label_lookup = {label: idx for idx, label in enumerate(labels)}

        # Whether to put whole epochs in index map only, or split it up into events
        # Will return (n_events, samples) probability distribution as labels
        self.whole_epoch = whole_epoch
        self.subset_cond = subset_cond
        self.statistics = statistics
        self.add_negative = add_negative
        self.probabilistic_labels = probabilistic_labels
        self.skip_samples = skip_samples
        self.cut_samples = cut_samples
        self.add_pe = add_pe
        self.subset_channels = subset_channels

        if self.transform is not None:
            for i, tf in enumerate(self.transform.transforms):
                if isinstance(tf, ConcatenateTransform):
                    self.concat_probability = tf.concat_probability
                    self.transform.transforms.pop(i)
                    break
            self.concat_probability = 0
        else:
            self.concat_probability = 0

        self.index_map = (
            self._create_index_map()
            if not self.whole_epoch
            else self._create_index_map_whole()
        )

        self.window_size = window_size
        if jiggle == 0:
            self.jiggle = lambda: 0
        else:
            self.jiggle = lambda: np.random.randint(-jiggle, jiggle)

        # Open first dataset to check if split
        with xr.open_dataset(data_paths[0]) as ds:
            self.split = "labels" not in ds.data_vars and "probabilities" not in ds
        if norm_vars is None:
            if self.statistics is None:
                self._calc_global_statistics(min(1000, self.__len__()))
            norm_vars = get_norm_vars_from_global_statistics(
                self.statistics, normalization_fn
            )

        self.normalization_fn = normalization_fn
        self.norm_vars = norm_vars

    def _create_index_map(self):
        index_map = []

        for file_idx, file_path in enumerate(self.data_paths):
            with xr.open_dataset(file_path) as ds:
                # Find trials for which all samples for one channel are NaN, these are excluded
                data = ds["data"].values
                # Select subset of data to check valid indices
                data = data[..., 0, :]
                mask_nan = ~np.isnan(data).all(axis=-1)

                probas = ds["probabilities"].values
                event_locs = probas.argmax(axis=-1)
                # Indices where at least one location is found
                mask_locs = event_locs.sum(axis=2) != 0

                if self.subset_cond is not None:
                    if self.subset_cond[1] == "contains":
                        subset_indices = ds[self.subset_cond[0]].str.contains(
                            self.subset_cond[2]
                        )
                    elif self.subset_cond[1] == "equal":
                        subset_indices = ds[self.subset_cond[0]] == self.subset_cond[2]
                    subset_indices = subset_indices.to_numpy()

                participant_indices = np.arange(event_locs.shape[0])[:, None, None]
                trial_indices = np.arange(event_locs.shape[1])[None, :, None]
                time_indices = event_locs

                nan_mask = np.isnan(
                    data[participant_indices, trial_indices, time_indices]
                )
                event_locs[nan_mask] = 0
                combined_indices = (
                    np.argwhere((mask_nan) & (mask_locs) & (subset_indices))
                    if self.subset_cond is not None
                    else np.argwhere((mask_nan) & (mask_locs))
                )

                if len(self.participants_to_keep) > 0:
                    participants_in_data = [
                        index
                        for index, value in enumerate(ds.participant.values.tolist())
                        if value in self.participants_to_keep
                    ]
                    index_map.extend(
                        [
                            (file_idx, *idx, loc, loc_idx)
                            for idx in combined_indices
                            if idx[0] in participants_in_data
                            for loc_idx, loc in enumerate(event_locs[idx[0], idx[1]])
                            if loc != 0
                        ]
                    )
                else:
                    index_map.extend(
                        [
                            (file_idx, *idx, loc, loc_idx)
                            for idx in combined_indices
                            for loc_idx, loc in enumerate(event_locs[idx[0], idx[1]])
                            if loc != 0
                        ]
                    )

        return index_map

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
            if not self.whole_epoch:
                label_counter[label] += 1

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

    def __getitem__(self, idx, concat=False, debug=False):
        indices = self.index_map[idx]

        ds = self._get_dataset(indices[0])
        n_samples = len(ds.samples)
        pad_left = 0
        pad_right = 0
        if self.whole_epoch:
            filter = {
                "participant": indices[1],
                "epochs": indices[2],
            }
        else:
            # Jiggle event idx to ensure that the model learns from samples where transition is not in the middle of the window
            event_idx = indices[3]
            event_idx += self.jiggle()

            min_sample = event_idx - self.window_size[0]
            max_sample = event_idx + self.window_size[1] + 1

            if min_sample < 0:
                pad_left = abs(min_sample)
                min_sample = 0
            if max_sample > n_samples:
                pad_right = max_sample - n_samples
                max_sample = n_samples
            filter = {
                "participant": indices[1],
                "epochs": indices[2],
                "samples": range(min_sample, max_sample),
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
        if not self.whole_epoch and self.split:
            sample_label = indices[4]
        elif self.split:
            sample_label = 0
        else:
            sample_label = torch.as_tensor(
                sample.probabilities.values, dtype=torch.float32
            )
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
        if debug:
            plot_epoch((sample_data, sample_label), "Raw")

        end_idx = get_masking_index(sample_data, search_value=torch.nan)
        sample_data[end_idx - self.cut_samples : end_idx, :] = torch.nan

        if not self.split:
            sample_label[end_idx - self.cut_samples : end_idx, :] = 0
            if self.add_negative:
                sample_label[end_idx - self.cut_samples : end_idx, 0] = 1.0
            sample_label = sample_label[self.skip_samples :, :]

        sample_data = sample_data[self.skip_samples :, :]

        if self.transform is not None:
            sample_data, sample_label = self.transform((sample_data, sample_label))
            if debug:
                plot_epoch((sample_data, sample_label), "Cut and transformed")
        if concat:
            return sample_data, sample_label
        elif self.concat_probability > 0:
            if torch.rand((1,)).item() < self.concat_probability:
                # Get random index (or make a pre-existing permutation so every concat is the same for every index?)
                # Need concat=true to prevent infinite loop with 1 proba
                concat_data, concat_label = self.__getitem__(
                    random.randint(0, self.__len__() - 1), concat=True
                )
                if self.transform is not None:
                    concat_data, concat_label = self.transform(
                        (concat_data, concat_label)
                    )

                end_idx = get_masking_index(sample_data, search_value=torch.nan)
                sample_data = torch.concat((sample_data[:end_idx], concat_data), dim=0)[
                    : self.max_length
                ]
                sample_label = torch.concat(
                    (sample_label[:end_idx], concat_label), dim=0
                )[: self.max_length]

                if debug:
                    plot_epoch((sample_data, sample_label), "Concat cut & transformed")

        sample_data = self.normalization_fn(sample_data, *self.norm_vars)

        # fillna with masking_value
        sample_data = torch.nan_to_num(sample_data, nan=MASKING_VALUE)

        # Add positional encoding
        if self.add_pe:
            sample_data, sample_label = add_relative_positional_encoding(
                (sample_data, sample_label)
            )
        if debug:
            plot_epoch((sample_data, sample_label), "End result")

        if self.keep_info:
            values_to_keep = [
                np.atleast_1d(sample[key].to_numpy()) for key in self.info_to_keep
            ]
            sample_info = [
                {key: value for key, value in zip(self.info_to_keep, values)}
                for values in zip(*values_to_keep)
            ]
            # For debugging
            # sample_info.append({"indices": indices})
            return sample_data, sample_label, sample_info
        return sample_data, sample_label

    def __getitem_clean__(self, idx):
        # For use in calculating normalization variables and class weights
        indices = self.index_map[idx]
        ds = self._get_dataset(indices[0])
        n_samples = len(ds.samples)

        pad_left = 0
        pad_right = 0
        if self.whole_epoch:
            filter = {
                "participant": indices[1],
                "epochs": indices[2],
            }
        else:
            # Jiggle event idx to ensure that the model learns from samples where transition is not in the middle of the window
            event_idx = indices[3] + self.jiggle()

            min_sample = event_idx - self.window_size[0]
            max_sample = event_idx + self.window_size[1] + 1

            if min_sample < 0:
                pad_left = abs(min_sample)
                min_sample = 0
            if max_sample > n_samples:
                pad_right = max_sample - n_samples
                max_sample = n_samples

            filter = {
                "participant": indices[1],
                "epochs": indices[2],
                "samples": range(min_sample, max_sample),
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
        if not self.whole_epoch:
            sample_label = indices[4]
        elif self.split:
            sample_label = 0
        else:
            sample_label = torch.as_tensor(
                sample.probabilities.values, dtype=torch.float32
            ).transpose(1, 0)

        return sample_data, sample_label

    def __len__(self):
        return len(self.index_map)

    def set_transform(self, compose: Compose):
        self.transform = compose
        for i, tf in enumerate(self.transform.transforms):
            if isinstance(tf, ConcatenateTransform):
                self.concat_probability = tf.concat_probability
                self.transform.transforms.pop(i)
                break
        self.concat_probability = 0
