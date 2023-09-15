import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard


earlyStopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=3,
)


def pad_to_max_sample_length(array: np.array, max_sample_length: int) -> np.array:
    """Pads ndarray to given length, in this case
    the length of the largest sample.

    Args:
        array (np.array): Array to be padded.
        max_sample_length (int): Length of largest sample.

    Returns:
        np.array: Padded array
    """
    padding = ((0, 0), (0, max_sample_length - array.shape[1]))
    return np.pad(array, padding)


def get_summary_str(model: tf.keras.Model) -> str:
    # Converts model summary from Keras to string, for logging to Tensorboard.
    lines = []
    model.summary(print_fn=lines.append)
    return "    " + "\n    ".join(lines)


# Credits:
# https://stackoverflow.com/questions/52453305/how-do-i-add-text-summary-to-tensorboard-on-keras
class LoggingTensorBoard(TensorBoard):
    def __init__(self, dict_to_log=None, **kwargs):
        super().__init__(**kwargs)
        self.dict_to_log = dict_to_log

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)

        writer = self._train_writer

        with writer.as_default():
            for key, value in self.dict_to_log.items():
                tf.summary.text(key, tf.convert_to_tensor(value), step=0)
