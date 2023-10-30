import random
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np

# Stops training if validation loss has not decreased for 3 epochs in a row, restoring the weights of the best epoch
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


def get_summary_str(model: tf.keras.Model) -> str:
    # Converts model summary from Keras to string, for logging to Tensorboard.
    lines = []
    model.summary(print_fn=lines.append)
    return "    " + "\n    ".join(lines)


def set_global_seed(seed: int) -> None:
    # Sets all (hopefully) random states used, to get more consistent training runs
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


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
