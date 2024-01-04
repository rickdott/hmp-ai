from torch.utils.data import Dataset
import xarray as xr
import torch
import numpy as np
from hmpai.data import SAT1_STAGES_ACCURACY, preprocess


class SAT1Dataset(Dataset):
    """
    PyTorch dataset for the SAT-1 dataset.

    Args:
        dataset (xr.Dataset): The input dataset.
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
    ):
        # Alphabetical ordering of labels used for categorization of labels
        label_lookup = {label: idx for idx, label in enumerate(labels)}

        # If labels is a data variable, the data is sequential instead of split
        sequential = dataset.data_vars.__contains__("labels")

        # Preprocess data
        if do_preprocessing:
            dataset = preprocess(
                dataset, shuffle=True, shape_topological=shape_topological, sequential=sequential
            )

        self.data = torch.as_tensor(dataset.data.to_numpy(), dtype=torch.float32)

        vectorized_label_to_index = np.vectorize(lambda x: label_lookup.get(x, -1))
        indices = xr.apply_ufunc(vectorized_label_to_index, dataset.labels)
        self.labels = torch.as_tensor(indices.values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
