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
    start_from_epoch=0,
)


def get_tensorboard_cb(path):
    return tf.keras.callbacks.TensorBoard(log_dir=path, histogram_freq=1)


def pad_to_max_sample_length(array, max_sample_length):
    padding = ((0, 0), (0, max_sample_length - array.shape[1]))
    return np.pad(array, padding)


def write_string_to_tb(writer, subject, string):
    with writer.as_default():
        tf.summary.text(subject, string, step=0)


def get_summary_str(model):
    lines = []
    model.summary(print_fn=lines.append)
    return '    ' + '\n    '.join(lines)


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