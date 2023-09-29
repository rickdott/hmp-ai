import tensorflow as tf
import numpy as np
import xbatcher
import xarray as xr
from shared.utilities import CHANNELS_2D, MASKING_VALUE
from shared.data import SAT1_STAGES_ACCURACY


class SAT1DataGenerator(tf.keras.utils.Sequence):
    sparse_height = CHANNELS_2D.shape[0]
    sparse_width = CHANNELS_2D.shape[1]

    def __init__(self, dataset: xr.Dataset, batch_size=16, shape_topological=False):
        # Alphabetical ordering of labels used for categorization of labels
        self.cat_labels = [label.item() for label in dataset.labels]

        self.batch_size = batch_size
        # Preprocess data
        # Stack three dimensions into one MultiIndex dimension 'index'
        dataset = dataset.stack({"index": ["participant", "epochs", "labels"]})
        # Reorder so index is in front and samples/channels are switched
        dataset = dataset.transpose("index", "samples", "channels")
        # Drop all indices for which all channels & samples are NaN, this happens in cases of
        # measuring error or label does not occur under condition in dataset
        dataset = dataset.dropna("index", how="all")
        dataset = dataset.fillna(MASKING_VALUE)

        # Reshape into (index, 8, 5, samples) sparse topological array
        if shape_topological:
            dataset = self.__reshape__(dataset)

        # Reshuffle
        np.random.seed(42)
        n = len(dataset.index)
        perm = np.random.permutation(n)
        dataset = dataset.isel(index=perm)

        # Get list of truth values, removing last to fit amount of batches
        self.full_labels = dataset.labels
        n_used = (n // batch_size) * batch_size
        self.full_labels = self.full_labels[:n_used]

        if shape_topological:
            input_dims = {
                "x": len(dataset.x),
                "y": len(dataset.y),
                "samples": len(dataset.samples),
            }
        else:
            input_dims = {
                "channels": len(dataset.channels),
                "samples": len(dataset.samples),
            }
        # Create xbatcher generator
        self.generator = xbatcher.BatchGenerator(
            dataset,
            input_dims=input_dims,
            batch_dims={"index": batch_size},
        )

    def __len__(self):
        return self.generator.__len__()

    def __getitem__(self, idx):
        batch = self.generator.__getitem__(idx)
        batch_labels = np.array(
            [self.cat_labels.index(label) for label in batch.labels]
        )
        return (batch.data, batch_labels)

    def __reshape__(self, dataset):
        # Create array full of 'empty' values (999)
        reshaped_data = np.full(
            (
                len(dataset.index),
                self.sparse_height,
                self.sparse_width,
                len(dataset.samples),
            ),
            MASKING_VALUE,
            dtype=dataset.data.dtype,
        )

        for x in range(self.sparse_width):
            for y in range(self.sparse_height):
                if CHANNELS_2D[y, x] == "NA":
                    continue
                # Set slice of reshaped data to be information from channel at position in CHANNELS_2D
                reshaped_data[:, y, x, :] = dataset.sel(channels=CHANNELS_2D[y, x]).data

        # Configure new dataset coordinates and assign the reshaped data
        new_coords = (
            dataset.coords.to_dataset()
            .drop_vars("channels")
            .assign_coords(
                {"x": np.arange(self.sparse_height), "y": np.arange(self.sparse_width)}
            )
        )
        dataset = new_coords.assign(
            data=(("index", "x", "y", "samples"), reshaped_data)
        )
        return dataset


class SequentialSAT1DataGenerator(tf.keras.utils.Sequence):
    # For models that use the whole sequence instead of split stages, IN PROGRESS
    def __init__(self, dataset: xr.Dataset, batch_size=16):
        # Alphabetical ordering of labels used for categorization of labels
        # should span all stages found in dataset
        self.cat_labels = SAT1_STAGES_ACCURACY

        self.batch_size = batch_size
        # Preprocess data
        # Stack three dimensions into one MultiIndex dimension 'index'
        dataset = dataset.stack({"index": ["participant", "epochs"]})
        # Reorder so that index is at the front
        dataset = dataset.transpose("index", "samples", "channels")
        # Drop all indices for which all channels & samples are NaN, this happens in cases of
        # measuring error or label does not occur under condition in dataset
        dataset = dataset.dropna("index", how="all", subset=["data"])
        dataset = dataset.fillna(MASKING_VALUE)

        # Reshuffle
        np.random.seed(42)
        n = len(dataset.index)
        perm = np.random.permutation(n)
        dataset = dataset.isel(index=perm)

        # Get list of truth values, removing last to fit amount of batches
        self.full_labels = dataset.labels
        n_used = (n // batch_size) * batch_size
        self.full_labels = self.full_labels[:n_used]

        input_dims = {
            "channels": len(dataset.channels),
            "samples": len(dataset.samples),
        }
        # Create xbatcher generator
        self.generator = xbatcher.BatchGenerator(
            dataset,
            input_dims=input_dims,
            batch_dims={"index": batch_size},
        )

    def __len__(self):
        return self.generator.__len__()

    def __getitem__(self, idx):
        batch = self.generator.__getitem__(idx)
        # Shape (16, 199) instead of (16, )
        batch_labels = np.array(
            [
                [self.cat_labels.index(label.item()) if label != "" else -1 for label in seq_labels]
                for seq_labels in batch.labels
            ]
        )
        return (batch.data, batch_labels)
