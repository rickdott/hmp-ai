import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard
import json

earlyStopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=3,
)

# Channel configuration for topological layout, "NA" means 'not available' and should not be used in training
CHANNELS_2D = np.array(
    [
        ["NA", "Fp1", "NA", "Fp2", "NA"],
        ["NA", "NA", "AFz", "NA", "NA"],
        ["F7", "F3", "Fz", "F4", "F8"],
        ["FC5", "FC1", "FCz", "FC2", "FC6"],
        ["T7", "C3", "Cz", "C4", "T8"],
        ["CP5", "CP1", "CPz", "CP2", "CP6"],
        ["P7", "P3", "Pz", "P4", "P8"],
        ["NA", "O1", "NA", "O2", "NA"],
    ],
    dtype=str,
)

# Value that means data should not be used in training
MASKING_VALUE = 999


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


def pretty_json(data: dict) -> str:
    # From https://www.tensorflow.org/tensorboard/text_summaries
    json_data = json.dumps(data, indent=2)
    return "".join(f"\t{line}" for line in json_data.splitlines(True))


def print_results(results: dict) -> str:
    # From a list of test results to an aggregated accuracy and F1-Score
    accuracy = 0.0
    f1 = 0.0
    for result in results:
        accuracy += result["accuracy"]
        f1 += result["macro avg"]["f1-score"]

    print(f'Average Accuracy: {accuracy / len(results)}')
    print(f'Average F1-Score: {f1 / len(results)}')


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
