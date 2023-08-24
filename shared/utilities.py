import tensorflow as tf
import numpy as np

earlyStopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)


def pad_to_max_sample_length(array, max_sample_length):
    padding = ((0, 0), (0, max_sample_length - array.shape[1]))
    return np.pad(array, padding)
