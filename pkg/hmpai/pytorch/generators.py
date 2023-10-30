from torch.utils.data import Dataset
import xarray as xr
import xbatcher
import torch
import numpy as np
from hmpai.data import SAT1_STAGES_ACCURACY, preprocess


# class EegDataset(Dataset):
#     def __init__(
#         self, dataset: xr.Dataset, shape_topological=False, do_preprocessing=True
#     ):
#         # Alphabetical ordering of labels used for categorization of labels
#         self.cat_labels = SAT1_STAGES_ACCURACY

#         # Preprocess data
#         if do_preprocessing:
#             dataset = preprocess(
#                 dataset, shuffle=True, shape_topological=shape_topological
#             )

#         # Get list of truth values
#         self.full_labels = dataset.labels

#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset.index)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         sample = self.dataset.isel(index=idx)
#         sample_label = torch.tensor(self.cat_labels.index(sample.labels))
#         return (torch.tensor(sample.data), sample_label)


class SAT1DataLoader():
    def __init__(
        self,
        dataset: xr.Dataset,
        batch_size=16,
        shape_topological=False,
        do_preprocessing=True,
    ):
        # Alphabetical ordering of labels used for categorization of labels
        self.cat_labels = SAT1_STAGES_ACCURACY

        self.batch_size = batch_size

        # Preprocess data
        if do_preprocessing:
            dataset = preprocess(
                dataset, shuffle=True, shape_topological=shape_topological
            )
        self.dataset = dataset
        # Get list of truth values, removing last to fit amount of batches
        self.full_labels = dataset.labels
        n_used = (len(dataset.index) // batch_size) * batch_size
        self.full_labels = self.full_labels[:n_used]

        if shape_topological:
            self.input_dims = {
                "x": len(dataset.x),
                "y": len(dataset.y),
                "samples": len(dataset.samples),
            }
        else:
            self.input_dims = {
                "channels": len(dataset.channels),
                "samples": len(dataset.samples),
            }
        # Create xbatcher generator
        self.generator = xbatcher.BatchGenerator(
            dataset,
            input_dims=self.input_dims,
            batch_dims={"index": self.batch_size},
        )

    def __len__(self):
        return self.generator.__len__()

    def __getitem__(self, idx):
        batch = self.generator.__getitem__(idx)
        batch_labels = np.array(
            [self.cat_labels.index(label) for label in batch.labels]
        )
        batch_data = batch.data.to_numpy().astype(np.float32)
        batch_data = batch_data[:, None, :, :]

        return (torch.tensor(batch_data), torch.tensor(batch_labels))

    def shuffle(self):
        n = len(self.dataset.index)
        perm = np.random.permutation(n)
        self.dataset = self.dataset.isel(index=perm)
        # Create xbatcher generator
        self.generator = xbatcher.BatchGenerator(
            self.dataset,
            input_dims=self.input_dims,
            batch_dims={"index": self.batch_size},
        )