import tensorflow as tf
import numpy as np

class SAT1DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size=16):
        # Shuffle data and labels
        perm = np.random.permutation(len(data))
        # TODO: Investigate if normalization is necessary
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        self.data = data[perm]
        self.labels = labels[perm]

        self.batch_size = batch_size
        self.categories = sorted(list(set(self.labels.flatten())))

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size, :, :, :]

        labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels = np.array([self.categories.index(label) for label in labels])

        # Shape (16, 30, 210), (, 16)
        return batch, labels