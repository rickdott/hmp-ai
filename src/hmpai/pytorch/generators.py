from torch.utils.data import Dataset
import xarray as xr
import torch
import numpy as np
from hmpai.data import SAT1_STAGES_ACCURACY, preprocess
from hmpai.pytorch.utilities import DEVICE


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
    ):
        self.interpolate_to = interpolate_to
        # Alphabetical ordering of labels used for categorization of labels
        label_lookup = {label: idx for idx, label in enumerate(labels)}

        # If labels is a data variable, the data is sequential instead of split
        sequential = dataset.data_vars.__contains__("labels")

        # Preprocess data
        if do_preprocessing:
            dataset = preprocess(
                dataset, shuffle=True, shape_topological=shape_topological, sequential=sequential
            )

        values_to_keep = [dataset[key].to_numpy() for key in info_to_keep]
        self.info = [dict(zip(info_to_keep, values)) for values in zip(*values_to_keep)]

        self.data = torch.as_tensor(dataset.data.to_numpy(), dtype=torch.float32)

        vectorized_label_to_index = np.vectorize(lambda x: label_lookup.get(x, -1))
        indices = xr.apply_ufunc(vectorized_label_to_index, dataset.labels)
        self.labels = torch.as_tensor(indices.values, dtype=torch.long)
        if set_to_zero:
            self.labels = torch.where(self.labels == -1, torch.tensor(0), self.labels)
        if interpolate_to != 0:
            self.data = torch.Tensor(self.__resample_batch_eeg__(self.data))
            self.labels = torch.Tensor(self.__resample_batch_labels__(self.labels)).long()
        if order_by_rt:
            if 'rt' not in info_to_keep:
                raise ValueError("rt must be included in info_to_keep and in the source data to be able to order by rt.")
            combined = list(zip(self.data, self.labels, self.info))
            sorted_combined = sorted(combined, key=lambda x: x[2]['rt'])
            self.data, self.labels, self.info = zip(*sorted_combined)
            self.data = torch.stack(self.data)
            self.labels = torch.stack(self.labels).long()
            self.info = list(self.info)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if len(self.info) > 0:
            return self.data[idx], self.labels[idx], self.info[idx]
        else:
            return self.data[idx], self.labels[idx]
        
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
                resampled_trials[i, :, j] = np.interp(target_time, original_time, trials[i, :, j])
        
        return resampled_trials


    def __resample_batch_labels__(self, labels):
        batch_size, original_length = labels.shape
    
        # Create the original and target time axes
        original_time = np.linspace(0, 1, original_length)
        target_time = np.linspace(0, 1, self.interpolate_to)
        
        # Prepare an array to hold the resampled labels
        resampled_labels = np.zeros((batch_size, self.interpolate_to),)
        
        # Vectorized interpolation
        resampled_labels = np.array([np.interp(target_time, original_time, labels[i]) for i in range(batch_size)], dtype=np.intc)
        
        return resampled_labels