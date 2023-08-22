import tensorflow as tf
import numpy as np


class SAT1DataGenerator(tf.keras.utils.Sequence):
    # Pass XArray dataset, already filtered
    def __init__(self, dataset, batch_size=16):
        # Dataset can contain NaN samples
        self.dataset = dataset
        # Normalizes to [0,1] TODO: Investigate if necessary or if [-1, 1] is better
        self.dataset = self.dataset.fillna(0)
        self.dataset = (self.dataset.data - np.min(self.dataset.data)) / (
            np.max(self.dataset.data) - np.min(self.dataset.data)
        )
        
        self.n_participants = len(dataset.participant)
        self.n_epochs = len(dataset.epochs)
        self.n_labels = len(dataset.labels)
        self.cat_labels = sorted(list(set(dataset.labels.to_numpy())))
        self.batch_size = batch_size

        # Find indices of NaN slices
        mask = self.dataset.isnull().all(dim=["channels", "samples"])
        nan_slices = np.argwhere(mask.data)
        cartesian = np.array(
            np.meshgrid(
                np.arange(self.n_participants),
                np.arange(self.n_epochs),
                np.arange(self.n_labels),
            )
        ).T.reshape(-1, 3)
        filtered_indices = np.array(
            [row for row in cartesian if row.tolist() not in nan_slices.tolist()]
        )
        self.n_valid_slices = filtered_indices.shape[0]

        # Shuffle valid indices
        np.random.seed(42)
        perm = np.random.permutation(self.n_valid_slices)
        self.filtered_indices = filtered_indices[perm]

    def __len__(self):
        return int(np.ceil(self.n_valid_slices / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.filtered_indices[
            idx * self.batch_size : (idx + 1) * self.batch_size, :
        ]
        dim1_indices = batch_indices[:, 0]
        dim2_indices = batch_indices[:, 1]
        dim3_indices = batch_indices[:, 2]

        batch_data = self.dataset[
            dim1_indices, dim2_indices, dim3_indices, slice(None), slice(None)
        ]
        batch_labels = batch_data.labels  # Labels as strings
        batch_labels = np.array([self.cat_labels.index(label) for label in batch_labels])

        # Returns data: (batch_size, channels, max sample length) and labels: (batch_size, )
        # return batch_data.data[0, 0, :, :, :], batch_labels
        return (batch_data.data[0, 0, :, :, :].reshape(-1, 30, 157, 1), batch_labels)
