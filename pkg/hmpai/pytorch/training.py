from hmpai.utilities import pretty_json
from tqdm.notebook import tqdm
from hmpai.pytorch.utilities import DEVICE, set_global_seed, get_summary_str
from hmpai.pytorch.generators import SAT1DataLoader
import torch
from hmpai.data import SAT1_STAGES_ACCURACY
from collections import Counter
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import xarray as xr
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report


def train_and_test(
    model: torch.nn.Module,
    train: xr.Dataset,
    test: xr.Dataset,
    val: xr.Dataset = None,
    batch_size: int = 16,
    epochs: int = 20,
    logs_path: Path = None,
    additional_info: dict = None,
    additional_name: str = None,
    loader: SAT1DataLoader = None,
    gen_kwargs: dict = None,
    use_class_weights: bool = True,
):
    set_global_seed(42)
    # Create loaders
    if loader is None:
        loader = SAT1DataLoader
    if gen_kwargs is None:
        gen_kwargs = dict()

    train_loader = loader(train, batch_size, **gen_kwargs)
    test_loader = loader(test, batch_size, **gen_kwargs)
    if val is not None:
        val_loader = loader(val, batch_size, **gen_kwargs)

    # Set up logging
    write_log = logs_path is not None
    if write_log:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if additional_name is not None:
            run_id = f"{run_id}_{additional_name}"
        path = logs_path / run_id
        writer = SummaryWriter(path)

        to_write = {
            "Model summary": get_summary_str(
                model,
                (
                    batch_size,
                    1,
                    train_loader.dataset.data.shape[1],
                    train_loader.dataset.data.shape[2],
                ),
            )
        }
        if additional_info:
            to_write.update(additional_info)

        for k, v in to_write.items():
            writer.add_text(k, v, global_step=0)

    # Set up optimizer and loss
    weight = (
        calculate_class_weights(train_loader).to(DEVICE)
        if use_class_weights
        else torch.ones((len(SAT1_STAGES_ACCURACY),))
    )
    loss = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.NAdam(model.parameters())
    stopper = EarlyStopper()

    for epoch in range(epochs):
        batch_losses = train(model, train_loader, opt, loss)

        # Shuffle data before next epoch
        train_loader.shuffle()

        val_losses, val_accuracy = validate(model, val_loader, loss)

        mean_train_loss = np.mean(batch_losses)
        mean_val_loss = np.mean(val_losses)

        writer.add_scalar("train_loss", mean_train_loss)
        writer.add_scalar("val_loss", mean_val_loss)
        writer.add_scalar("val_accuracy", val_accuracy)

        print(
            f"Epoch {epoch}, loss: {mean_train_loss}, val_loss: {mean_val_loss}, val_accuracy: {val_accuracy}"
        )

        # Stop training if validation loss has not improved sufficiently
        if stopper.check_stop(mean_val_loss):
            break


def train(
    model: torch.nn.Module,
    train_loader: SAT1DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.modules.loss._Loss,
) -> list[float]:
    """Train model for all batches, one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (SAT1DataLoader): Loader that contains data used.
        optimizer (torch.optim.Optimizer): Optimizer used.
        loss_function (torch.nn.modules.loss._Loss): Loss function used.

    Returns:
        list[float]: List containing loss for each batch.
    """
    model.train(True)

    loss_per_batch = []

    for batch in tqdm(train_loader):
        # (Index, samples, channels), (Index, )
        data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

        optimizer.zero_grad()

        predictions = model(data)

        loss = loss_function(predictions, labels)
        loss_per_batch.append(loss.item())

        loss.backward()

        optimizer.step()

    return loss_per_batch


def validate(
    model: torch.nn.Module,
    validation_loader: SAT1DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
) -> (list[float], float):
    """Validate model.

    Args:
        model (torch.nn.Module): Model to validate.
        validation_loader (SAT1DataLoader): Loader containing validation data.
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
        for batch in tqdm(validation_loader):
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
    model: torch.nn.Module, test_loader: SAT1DataLoader, writer: SummaryWriter
) -> dict:
    outputs = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            data = batch[0].to(DEVICE)

            predictions = model(data)
            predicted_labels = torch.argmax(predictions, dim=1)
            outputs.append(predicted_labels)

    predicted_classes = [
        test_loader.cat_labels[idx] for idx in torch.cat(outputs, dim=0)
    ]

    test_results = classification_report(
        test_loader.full_labels, predicted_classes, output_dict=True
    )

    writer.add_text("Test results", pretty_json(test_results), global_step=0)

    return test_results


def calculate_class_weights(generator) -> torch.Tensor:
    counter = Counter(generator.full_labels.to_numpy())
    total = sum(counter.values())
    weights = []
    for stage in SAT1_STAGES_ACCURACY:
        weights.append(total / counter[stage])
    return torch.FloatTensor(weights)


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
            if self.counter >= self.patience:
                return True
        return False
