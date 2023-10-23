import tensorflow as tf
import numpy as np
import xbatcher
import xarray as xr
from shared.utilities import CHANNELS_2D
from shared.data import SAT1_STAGES_ACCURACY, preprocess


class SAT1DataGenerator(tf.keras.utils.Sequence):
    sparse_height = CHANNELS_2D.shape[0]
    sparse_width = CHANNELS_2D.shape[1]

    def __init__(self, dataset: xr.Dataset, batch_size=16, shape_topological=False, do_preprocessing=True):
        # Alphabetical ordering of labels used for categorization of labels
        self.cat_labels = SAT1_STAGES_ACCURACY

        self.batch_size = batch_size

        # Preprocess data
        if do_preprocessing:
            dataset = preprocess(dataset, shuffle=True, shape_topological=shape_topological)

        # Get list of truth values, removing last to fit amount of batches
        self.full_labels = dataset.labels
        n_used = (len(dataset.index) // batch_size) * batch_size
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


class SequentialSAT1DataGenerator(tf.keras.utils.Sequence):
    # For models that use the whole sequence instead of split stages, IN PROGRESS
    def __init__(self, dataset: xr.Dataset, batch_size=16, do_preprocessing=True):
        # Alphabetical ordering of labels used for categorization of labels
        # should span all stages found in dataset
        self.cat_labels = SAT1_STAGES_ACCURACY

        self.batch_size = batch_size
        # Preprocess data
        if do_preprocessing:
            dataset = preprocess(dataset, shuffle=True, sequential=True)

        # Get list of truth values, removing last to fit amount of batches
        self.full_labels = dataset.labels
        n_used = (len(dataset.index) // batch_size) * batch_size
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
                [
                    self.cat_labels.index(label.item()) if label != "" else 999
                    for label in seq_labels
                ]
                for seq_labels in batch.labels
            ]
        )
        return (batch.data, batch_labels)
