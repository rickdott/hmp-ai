from shared.generators import SAT1DataGenerator
import random
import datetime
from shared.utilities import get_summary_str, earlyStopping_cb
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import xarray as xr
from keras.callbacks import TensorBoard
from typing import Callable
from shared.normalization import norm_0_to_1

def k_fold_cross_validate(data: xr.Dataset,
                          k: int,
                          normalization_fn: Callable[[xr.Dataset, float, float], xr.Dataset] = norm_0_to_1):
    # Make sure #participants is divisible by k
    # Divide data into k folds
    # For each fold, train model (log?)
    # Gather accuracy/F1 from sklearn classification_report
    # Return {folds: scores{metric: value}} dictionary?
    pass

def split_data_on_participants(
    data: xr.Dataset,
    train_percentage: int = 60,
    normalization_fn: Callable[[xr.Dataset, float, float], xr.Dataset] = norm_0_to_1,
) -> (xr.Dataset, xr.Dataset, xr.Dataset):
    """Splits dataset into three distinct set on participant, ensuring
    that no participant occurs in more than one sets.
    Splits remainder of train percentage into two sets.
    Also normalizes data based on training set parameters to prevent information leakage.

    Args:
        data (xr.Dataset): Dataset to be split.
        train_percentage (int): Percentage of participants used in the training set. Defaults to 60.

    Returns:
        (xr.Dataset, xr.Dataset, xr.Dataset): tuple of train, test, val datasets.
    """
    random.seed(42)
    participants = data.participant.values.tolist()
    # In case of SAT1 experiment, 25 participants are used
    # Given train_percentage=60, remaining 40 percent will be split evenly between validation and test sets
    # 100-train_percentage must be divisible by 2

    # Find amounts of train and test/val participants
    train_n = int(len(participants) * (train_percentage / 100))
    testval_n = len(participants) - train_n

    # Split into train, test, and val by sampling randomly
    testval_participants = random.sample(participants, testval_n)
    train_participants = [p for p in participants if p not in testval_participants]
    val_participants = testval_participants[: testval_n // 2]
    test_participants = testval_participants[testval_n // 2 :]

    # Select subsets from data
    train_data = data.sel(participant=train_participants)
    val_data = data.sel(participant=val_participants)
    test_data = data.sel(participant=test_participants)

    # Normalize data
    train_min = train_data.min(skipna=True).data.item()
    train_max = train_data.max(skipna=True).data.item()

    train_data = normalization_fn(train_data, train_min, train_max)
    val_data = normalization_fn(val_data, train_min, train_max)
    test_data = normalization_fn(test_data, train_min, train_max)

    return train_data, val_data, test_data


def train_and_evaluate(
    model: tf.keras.Model,
    train: xr.Dataset,
    val: xr.Dataset,
    test: xr.Dataset,
    batch_size: int = 16,
    epochs: int = 20,
    workers: int = 8,
    logs_path: Path = Path("logs/"),
    additional_info: dict = None,
) -> tf.keras.callbacks.History:
    """Trains and evaluates a given model on the given datasets.
    After training the model is tested on the test set, results are logged to Tensorboard.

    Args:
        model (tf.keras.Model): Model to be used in training.
        train (xr.Dataset): Training dataset.
        val (xr.Dataset): Valuation dataset.
        test (xr.Dataset): Testing dataset.
        batch_size (int, optional): Batch size used when training the model. Defaults to 16.
        epochs (int, optional): How many epochs the model should train for. Defaults to 20.
        workers (int, optional): How many workers (CPU threads) should be used in training. Defaults to 8.
        additional_info (dict, optional): Additional info to be logged to Tensorboard. Defaults to None.

    Returns:
        tf.keras.History: History of model fitting, detailing loss/accuracy.
    """
    # Create generators
    train_gen = SAT1DataGenerator(train, batch_size)
    val_gen = SAT1DataGenerator(val, batch_size)
    test_gen = SAT1DataGenerator(test, batch_size)

    # Set up configuration logging
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = logs_path / run_id
    to_write = {"Model summary": get_summary_str(model)}
    if additional_info:
        to_write.update(additional_info)

    use_multiprocessing = workers != 1
    fit = model.fit(
        train_gen,
        epochs=epochs,
        callbacks=[earlyStopping_cb, LoggingTensorBoard(to_write, log_dir=path)],
        validation_data=val_gen,
        use_multiprocessing=use_multiprocessing,
        workers=workers,
    )

    # Test model and write test summary
    writer = tf.summary.create_file_writer(str(path / "train"))
    predicted_classes = np.argmax(model.predict(test_gen), axis=1)
    predicted_classes = [test_gen.cat_labels[idx] for idx in list(predicted_classes)]
    test_results = classification_report(
        test_gen.full_labels, predicted_classes
    ).splitlines()
    test_results = "    " + "\n    ".join(test_results)

    with writer.as_default():
        tf.summary.text("Test results", test_results, step=0)

    return fit


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
