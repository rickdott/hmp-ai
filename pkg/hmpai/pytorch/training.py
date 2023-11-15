from hmpai.utilities import pretty_json
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from hmpai.pytorch.utilities import (
    DEVICE,
    set_global_seed,
    get_summary_str,
    save_model,
    load_model,
)
from hmpai.pytorch.generators import SAT1Dataset
import torch
from hmpai.data import SAT1_STAGES_ACCURACY
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import xarray as xr
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report
from typing import Callable
from hmpai.normalization import norm_dummy
from copy import deepcopy
import re


def train_and_test(
    model: torch.nn.Module,
    train_set: Dataset,
    test_set: Dataset,
    val_set: Dataset = None,
    batch_size: int = 128,
    epochs: int = 20,
    workers: int = 4,
    logs_path: Path = None,
    additional_info: dict = None,
    additional_name: str = None,
    use_class_weights: bool = True,
    seed: int = 42,
) -> dict:
    """
    Trains and tests a PyTorch model on the given datasets.

    Args:
        model (torch.nn.Module): The PyTorch model to train and test.
        train_set (Dataset): The training dataset.
        test_set (Dataset): The testing dataset.
        val_set (Dataset, optional): The validation dataset. Defaults to None.
        batch_size (int, optional): The batch size to use for training and testing. Defaults to 128.
        epochs (int, optional): The number of epochs to train for. Defaults to 20.
        workers (int, optional): The number of worker threads to use for loading data. Defaults to 4.
        logs_path (Path, optional): The path to save logs to. Defaults to None.
        additional_info (dict, optional): Additional information to log. Defaults to None.
        additional_name (str, optional): Additional name to append to the log directory. Defaults to None.
        use_class_weights (bool, optional): Whether to use class weights for the loss function. Defaults to True.
        seed (int, optional): The seed to use for reproducibility. Defaults to 42.

    Returns:
        dict: A dictionary containing the test results.
    """
    set_global_seed(seed)

    # Create loaders
    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    # Do not shuffle test loader since testing should be the same always
    test_loader = DataLoader(
        test_set, batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    if val_set is not None:
        val_loader = DataLoader(
            val_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )

    # Set up logging
    write_log = logs_path is not None
    if write_log:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if additional_name is not None:
            run_id = f"{run_id}_{additional_name}"
        path = logs_path / run_id
        writer = SummaryWriter(path)

        # Log model summary
        shape = list(train_loader.dataset.data.shape)
        shape[0] = batch_size
        to_write = {"Model summary": get_summary_str(model, shape)}
        if additional_info:
            to_write.update(additional_info)

        for k, v in to_write.items():
            writer.add_text(k, v, global_step=0)

    # Set up optimizer and loss
    weight = (
        calculate_class_weights(train_set).to(DEVICE)
        if use_class_weights
        else torch.ones((len(SAT1_STAGES_ACCURACY),))
    )
    loss = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.NAdam(model.parameters())
    stopper = EarlyStopper()

    lowest_mean_val_loss = np.inf
    for epoch in range(epochs):
        with tqdm(total=len(train_loader), unit=" batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")

            # Train on batches in train_loader
            batch_losses = train(model, train_loader, opt, loss, tepoch)

            # Validate model and communicate results
            val_losses, val_accuracy = validate(model, val_loader, loss)
            tepoch.set_postfix(
                {
                    "loss": np.mean(batch_losses),
                    "val_loss": np.mean(val_losses),
                    "val_accuracy": val_accuracy,
                }
            )
            mean_train_loss = np.mean(batch_losses)
            mean_val_loss = np.mean(val_losses)

            # Save model checkpoint if validation loss is the lowest yet
            if mean_val_loss < lowest_mean_val_loss:
                lowest_mean_val_loss = mean_val_loss
                if write_log:
                    save_model(
                        path / "checkpoint.pt",
                        epoch,
                        model.state_dict(),
                        opt.state_dict(),
                        loss,
                    )
            writer.add_scalar("loss", mean_train_loss, global_step=epoch)
            writer.add_scalar("val_loss", mean_val_loss, global_step=epoch)
            writer.add_scalar("val_accuracy", val_accuracy, global_step=epoch)
            writer.flush()

            # Stop training if validation loss has not improved sufficiently
            if stopper.check_stop(mean_val_loss):
                break

    # Re-load best performing model
    if write_log:
        best_checkpoint = load_model(path / "checkpoint.pt")
        model.load_state_dict(best_checkpoint["model_state_dict"])
        opt.load_state_dict(best_checkpoint["optimizer_state_dict"])
        epoch = best_checkpoint["epoch"]
        loss = best_checkpoint["loss"]

    # Test model
    results = test(model, test_loader, writer)
    return results


def k_fold_cross_validate(
    model: torch.nn.Module,
    model_kwargs: dict,
    data: xr.Dataset,
    k: int,
    batch_size: int = 128,
    epochs: int = 20,
    normalization_fn: Callable[[xr.Dataset, float, float], xr.Dataset] = norm_dummy,
    gen_kwargs: dict = None,
    train_kwargs: dict = None,
    seed: int = 42,
) -> list[dict]:
    """
    Performs k-fold cross-validation on a given PyTorch model using the provided data.

    Args:
        model (torch.nn.Module): The PyTorch model to train and test.
        model_kwargs (dict): The keyword arguments to pass to the model constructor.
        data (xr.Dataset): The dataset to use for training and testing.
        k (int): The number of folds to use for cross-validation.
        batch_size (int, optional): The batch size to use for training and testing. Defaults to 128.
        epochs (int, optional): The number of epochs to train the model for. Defaults to 20.
        normalization_fn (Callable[[xr.Dataset, float, float], xr.Dataset], optional):
            The function to use for normalizing the data. Defaults to norm_dummy.
        gen_kwargs (dict, optional): The keyword arguments to pass to the SAT1Dataset constructor. Defaults to None.
        train_kwargs (dict, optional): The keyword arguments to pass to the train_and_test function. Defaults to None.
        seed (int, optional): The seed to use for reproducibility. Defaults to 42.

    Returns:
        list[dict]: A list of dictionaries containing the results of each fold.
    """
    if gen_kwargs is None:
        gen_kwargs = dict()
    results = []
    set_global_seed(seed)
    folds = get_folds(data, k)
    for i_fold in range(len(folds)):
        # Deepcopy since folds is changed in memory when .pop() is used, and folds needs to be re-used
        train_folds = deepcopy(folds)
        test_fold = train_folds.pop(i_fold)
        train_fold = np.concatenate(train_folds, axis=0)
        print(f"Fold {i_fold + 1}: test fold: {test_fold}")

        train_data = data.sel(participant=train_fold)
        test_data = data.sel(participant=test_fold)

        # Normalize data
        train_min = train_data.min(skipna=True).data.item()
        train_max = train_data.max(skipna=True).data.item()
        train_data = normalization_fn(train_data, train_min, train_max)
        test_data = normalization_fn(test_data, train_min, train_max)

        train_dataset = SAT1Dataset(train_data, **gen_kwargs)
        test_dataset = SAT1Dataset(test_data, **gen_kwargs)

        # Resets model every fold
        model_instance = model(**model_kwargs).to(DEVICE)

        # Add fold info to name to compare conditions across folds
        if train_kwargs["additional_name"]:
            additional_name = train_kwargs["additional_name"]
            if "fold" in additional_name:
                additional_name = re.sub(
                    "_fold[0-9]*", f"_fold{i_fold + 1}", additional_name
                )
            else:
                additional_name = additional_name + f"_fold{i_fold + 1}"
            train_kwargs["additional_name"] = additional_name

        # Train and test model
        result = train_and_test(
            model_instance,
            train_dataset,
            test_dataset,
            val_set=test_dataset,
            batch_size=batch_size,
            epochs=epochs,
            **train_kwargs,
        )
        print(f"Fold {i_fold + 1}: Accuracy: {result['accuracy']}")
        print(f"Fold {i_fold + 1}: F1-Score: {result['macro avg']['f1-score']}")
        results.append(result)

    return results


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
    progress: tqdm = None,
) -> list[float]:
    """Train model for all batches, one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Loader that contains data used.
        optimizer (torch.optim.Optimizer): Optimizer used.
        loss_function (torch.nn.modules.loss._Loss): Loss function used.
        progress (tqdm, optional): tqdm instance to write progress to, will not write if not provided. Defaults to None.

    Returns:
        list[float]: List containing loss for each batch.
    """
    model.train()

    loss_per_batch = []

    for i, batch in enumerate(train_loader):
        # (Index, samples, channels), (Index, )
        data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

        optimizer.zero_grad()

        predictions = model(data)

        loss = loss_function(predictions, labels)
        loss_per_batch.append(loss.item())

        # Update loss shown every 5 batches, otherwise it is illegible
        if progress is not None:
            progress.update(1)
            if i % 5 == 0:
                progress.set_postfix({"loss": round(np.mean(loss_per_batch), 5)})

        loss.backward()

        optimizer.step()

    return loss_per_batch


def validate(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
) -> (list[float], float):
    """Validate model.

    Args:
        model (torch.nn.Module): Model to validate.
        validation_loader (DataLoader): Loader containing validation data.
        loss_function (torch.nn.modules.loss._Loss): Loss function used.

    Returns:
        list[float]: List containing loss for each batch.
        float: Validation accuracy
    """
    model.eval()

    loss_per_batch = []
    total_correct = 0
    total_instances = 0

    with torch.no_grad():
        for batch in validation_loader:
            # (Index, samples, channels), (Index, )
            data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            predictions = model(data)
            predicted_labels = torch.argmax(predictions, dim=1)
            correct_predictions = sum(predicted_labels == labels).item()

            total_correct += correct_predictions
            total_instances += len(labels)

            loss = loss_function(predictions, labels)
            loss_per_batch.append(loss.item())

    return loss_per_batch, round(total_correct / total_instances, 5)


def test(
    model: torch.nn.Module, test_loader: DataLoader, writer: SummaryWriter
) -> dict:
    """
    Test the PyTorch model on the given test data and return the classification report.

    Args:
        model (torch.nn.Module): The PyTorch model to test.
        test_loader (DataLoader): The DataLoader containing the test data.
        writer (SummaryWriter): The SummaryWriter to use for logging.

    Returns:
        dict: The classification report as a dictionary.
    """
    model.eval()
    outputs = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            data, labels = batch[0].to(DEVICE), batch[1]

            predictions = model(data)
            predicted_labels = torch.argmax(predictions, dim=1)
            outputs.append(predicted_labels)
            true_labels.append(labels)

    predicted_classes = torch.cat(outputs, dim=0)
    true_classes = torch.cat(true_labels, dim=0)
    test_results = classification_report(
        true_classes, predicted_classes.cpu(), output_dict=True
    )

    writer.add_text("Test results", pretty_json(test_results), global_step=0)

    return test_results


def calculate_class_weights(set: torch.utils.data.Dataset) -> torch.Tensor:
    """
    Calculates class weights for a given dataset.

    Args:
        set (torch.utils.data.Dataset): The dataset to calculate class weights for.

    Returns:
        torch.Tensor: The calculated class weights.
    """
    occurrences = set.labels.unique(return_counts=True)
    weights = sum(occurrences[1]) / occurrences[1]
    return weights


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
    np.random.shuffle(participants)
    folds = np.array_split(participants, k)
    return folds


# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, tolerance=3, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def check_stop(self, validation_loss):
        # Returns True if the model has not improved for <tolerance> epochs
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        return False
