from torch.utils.data import Dataset
import xarray as xr
import torch
from hmpai.data import SAT1_STAGES_ACCURACY, preprocess


class SAT1Dataset(Dataset):
    def __init__(
        self, dataset: xr.Dataset, shape_topological=False, do_preprocessing=True
    ):
        # Alphabetical ordering of labels used for categorization of labels
        self.cat_labels = SAT1_STAGES_ACCURACY

        # Preprocess data
        if do_preprocessing:
            dataset = preprocess(
                dataset, shuffle=True, shape_topological=shape_topological
            )
        self.data = torch.as_tensor(dataset.data.to_numpy(), dtype=torch.float32)[
            :, None, :, :
        ]
        self.labels = torch.as_tensor(
            [self.cat_labels.index(label) for label in dataset.labels]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
