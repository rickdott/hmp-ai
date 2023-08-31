import tensorflow as tf
import numpy as np
import xbatcher


class SAT1DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=16):
        # Alphabetical ordering of labels used for categorization of labels
        self.cat_labels = [label.item() for label in dataset.labels]

        self.batch_size = batch_size
        # Preprocess data
        # Stack three dimensions into one MultiIndex dimension 'index'
        dataset = dataset.stack({"index": ["participant", "epochs", "labels"]})
        # Reorder so that index is at the front
        dataset = dataset.transpose("index", ...)
        # Drop all indices for which all channels & samples are NaN, this happens in cases of
        # measuring error or label does not occur under condition in dataset
        dataset = dataset.dropna("index", how="all")
        dataset = dataset.fillna(0)

        # Normalize to [0,1] TODO: Investigate if necessary or if [-1, 1] is better
        dataset = (dataset.data - np.min(dataset.data)) / (
            np.max(dataset.data) - np.min(dataset.data)
        )

        # Reshuffle
        np.random.seed(42)
        n = len(dataset.index)
        perm = np.random.permutation(n)
        dataset = dataset[perm]

        # Get list of truth values, removing last to fit amount of batches
        self.full_labels = dataset.labels
        n_used = (n // batch_size) * batch_size
        self.full_labels = self.full_labels[:n_used]

        # Create xbatcher generator
        self.generator = xbatcher.BatchGenerator(
            dataset,
            input_dims={
                "channels": len(dataset.channels),
                "samples": len(dataset.samples),
            },
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
