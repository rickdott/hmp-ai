from shared.generators import SAT1DataGenerator
import random
import datetime
from shared.utilities import get_summary_str, earlyStopping_cb, LoggingTensorBoard
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import xarray as xr
from typing import Callable
from shared.normalization import norm_0_to_1
from copy import deepcopy


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
    test: xr.Dataset,
    val: xr.Dataset = None,
    batch_size: int = 16,
    epochs: int = 20,
    workers: int = 8,
    logs_path: Path = None,
    additional_info: dict = None,
    additional_name: str = None,
    generator: tf.keras.utils.Sequence = None,
    gen_kwargs: dict = None,
) -> (tf.keras.callbacks.History, dict):
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
        logs_path (Path, optional): If given, log info to TensorBoard at given Path, if excluded, return testing result.
        additional_info (dict, optional): Additional info to be logged to Tensorboard. Defaults to None.
        additional_name (str, optional): Additional text to be added to the run name. Defaults to None.
        generator (tf.keras.utils.Sequence, optional): Which generator class to use. Defaults to SAT1DataGenerator
        gen_kwargs (dict, optional): Extra arguments for the generator. Defaults to None.

    Returns:
        tf.keras.History: History of model fitting, detailing loss/accuracy.
        dict: Result of test run, only given if logs_path is None
    """
    # Create generators
    if generator is None:
        generator = SAT1DataGenerator
    if gen_kwargs is None:
        gen_kwargs = dict()
    train_gen = generator(train, batch_size, **gen_kwargs)
    if val is not None:
        val_gen = generator(val, batch_size, **gen_kwargs)
    else:
        val_gen = None
    test_gen = generator(test, batch_size, **gen_kwargs)

    callbacks = []
    if val_gen is not None:
        callbacks.append(earlyStopping_cb)

    # Set up configuration logging
    write_log = logs_path is not None
    if write_log:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if additional_name is not None:
            run_id = run_id + '_' + additional_name
        path = logs_path / run_id

        to_write = {"Model summary": get_summary_str(model)}
        if additional_info:
            to_write.update(additional_info)
        callbacks.append(LoggingTensorBoard(to_write, log_dir=path))

    use_multiprocessing = workers != 1
    fit = model.fit(
        train_gen,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_gen,
        use_multiprocessing=use_multiprocessing,
        workers=workers,
    )

    # Test model and write test summary
    test_args = {}
    if write_log:
        test_args["logs_path"] = path
    test_results = test_model(model, test_gen, log_report=write_log, **test_args)

    return fit, test_results


def test_model(
    model: tf.keras.Model,
    test_gen: tf.keras.utils.Sequence,
    logs_path: Path = None,
    log_report: bool = False,
) -> dict | None:
    """Tests a given model, returning test report as dictionary if log_report is False,
    otherwise write test report to TensorBoard logs

    Args:
        model (tf.keras.Model): Model to be tested
        test_gen (tf.keras.utils.Sequence): Generator containing test data
        logs_path (Path, optional): Path for logging
        log_report (bool, optional): Whether to log the report to TensorBoard. Defaults to False.

    Raises:
        ValueError: If log_report is True, logs_path must be provided

    Returns:
        dict | None: Returns dict if log_report is False, otherwise None
    """
    if log_report:
        if logs_path is None:
            raise ValueError(
                "If log_report is set to True, the parameter logs_path must be provided"
            )
        else:
            writer = tf.summary.create_file_writer(str(logs_path / "train"))
    predicted_classes = np.argmax(model.predict(test_gen), axis=1)
    predicted_classes = [test_gen.cat_labels[idx] for idx in list(predicted_classes)]
    if log_report:
        test_results = classification_report(test_gen.full_labels, predicted_classes)
        test_results_str = "    " + "\n    ".join(test_results.splitlines())
        with writer.as_default():
            tf.summary.text("Test results", test_results_str, step=0)
        return
    else:
        test_results = classification_report(
            test_gen.full_labels, predicted_classes, output_dict=True
        )
        return test_results


def k_fold_cross_validate(
    data: xr.Dataset,
    model: tf.keras.Model,
    k: int,
    batch_size: int = 16,
    epochs: int = 10,
    workers: int = 8,
    normalization_fn: Callable[[xr.Dataset, float, float], xr.Dataset] = norm_0_to_1,
    gen_kwargs: dict = None,
) -> list[dict]:
    """Validate model performance using K-fold Cross Validation.

    Args:
        data (xr.Dataset): Data used for training the model.
        model (tf.keras.Model): Model to be evaluated.
        k (int): Amount of folds to be made, must divide the amount of participants in the dataset evenly.
        batch_size (int, optional): Batch size. Defaults to 16.
        epochs (int, optional): Amount of epochs to train for. Defaults to 10.
        workers (int, optional): Number of workers used in multiprocessing. Defaults to 8.
        normalization_fn (Callable[[xr.Dataset, float, float], xr.Dataset], optional): Normalization function to use. Defaults to norm_0_to_1.
        gen_kwargs (dict, optional): Optional arguments for the generator to pass on to train_and_evaluate.

    Returns:
        list[dict]: List of dictionaries detailing model performance per-class and averag.
    """

    results = []
    folds = get_folds(data, k)
    # Get first initialization of weights to return to later and compare equally
    weights = model.get_weights()

    for i in range(len(folds)):
        # Deepcopy since folds is changed in memory when .pop() is used, and folds needs to be re-used
        train_folds = deepcopy(folds)
        test_fold = train_folds.pop(i)
        train_fold = np.concatenate(train_folds, axis=0)
        print(f"Fold {i + 1}: test fold: {test_fold}")

        train_data = data.sel(participant=train_fold)
        test_data = data.sel(participant=test_fold)

        # Normalize data
        train_min = train_data.min(skipna=True).data.item()
        train_max = train_data.max(skipna=True).data.item()
        train_data = normalization_fn(train_data, train_min, train_max)
        test_data = normalization_fn(test_data, train_min, train_max)

        # Train model and test
        result = train_and_evaluate(
            model,
            train_data,
            test_data,
            batch_size=batch_size,
            epochs=epochs,
            workers=workers,
            gen_kwargs=gen_kwargs,
        )

        # Add test results to list
        print(f"Fold {i + 1}: accuracy: {result[1]['accuracy']}")
        results.append(result[1])
        model.set_weights(weights)
    return results


def get_folds(
    data: xr.Dataset,
    k: int,
) -> list[np.ndarray]:
    """Divides dataset into folds

    Args:
        data (xr.Dataset): Dataset to be used
        k (int): Amount of folds

    Raises:
        ValueError: Occurs when k does not divide number of participants

    Returns:
        list[np.ndarray]: List of folds
    """
    # Make sure #participants is divisible by k
    n_participants = len(data.participant)
    if n_participants % k != 0:
        raise ValueError(
            f"K: {k} (amount of folds) must divide number of participants: {n_participants}"
        )

    # Divide data into k folds
    participants = data.participant.values.copy()
    np.random.seed(42)
    np.random.shuffle(participants)
    folds = np.array_split(participants, k)
    return folds
