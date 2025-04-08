from collections import defaultdict, Counter
from hmpai.pytorch.correlation import correlate, emd
from hmpai.pytorch.normalization import get_norm_vars_from_global_statistics
from hmpai.utilities import pretty_json
from hmpai.visualization import plot_loss
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from hmpai.pytorch.utilities import (
    DEVICE,
    set_global_seed,
    save_model,
    load_model,
)
from hmpai.pytorch.generators import MultiXArrayProbaDataset, SAT1Dataset
from hmpai.pytorch.pretraining import *
import torch
from hmpai.data import SAT1_STAGES_ACCURACY, SAT_CLASSES_ACCURACY
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import netCDF4
import xarray as xr
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report
from typing import Callable
from hmpai.normalization import get_norm_vars, norm_dummy
from copy import deepcopy
import re
from hmpai.training import get_folds, split_participants


def train_and_test(
    model: torch.nn.Module,
    train_set: Dataset,
    test_set: Dataset | list[Dataset],
    val_set: Dataset = None,
    batch_size: int = 128,
    epochs: int = 20,
    workers: int = 4,
    logs_path: Path = None,
    additional_info: dict = None,
    additional_name: str = None,
    use_class_weights: bool = True,
    class_weights: torch.Tensor = None,
    label_smoothing: float = 0.0,
    weight_decay: float = 0.0,
    lr: float = 0.002,  # Default learning rate for optimizer
    do_spectral_decoupling: bool = False,
    labels: list[str] = None,
    seed: int = 42,
    pretrain_fn: Callable = None,
    whole_epoch: bool = False,
    probabilistic_labels: bool = False,
    do_test_shuffled: bool = False,
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
        label_smoothing (float, optional): The amount of label smoothing to use. Defaults to 0.0.
        weight_decay (float, optional): The amount of weight decay to use. Defaults to 0.0.
        do_spectral_decoupling (bool, optional): Whether to apply spectral decoupling to the loss function. Defaults to False.
        labels (list[str], optional): The labels to use for classification. Defaults to None.
        seed (int, optional): The seed to use for reproducibility. Defaults to 42.

    Returns:
        dict: A dictionary containing the test results.
    """
    set_global_seed(seed)
    torch.cuda.empty_cache()
    # Create loaders
    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    # Do not shuffle test loader since testing should be the same always
    test_loaders = []
    if type(test_set) is list:
        test_loaders = []
        for test_data in test_set:
            test_loaders.append(
                DataLoader(
                    test_data,
                    batch_size,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                )
            )
    else:
        # Assume type of test_set is Dataset
        test_loaders.append(
            DataLoader(
                test_set,
                batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
            )
        )

    val_loaders = []
    if val_set is not None:
        if type(val_set) is list:
            for val in val_set:
                val_loaders.append(
                    DataLoader(
                        val,
                        batch_size,
                        shuffle=True,
                        num_workers=workers,
                        pin_memory=True,
                    )
                )
        else:
            val_loaders.append(
                DataLoader(
                    val_set,
                    batch_size,
                    shuffle=True,
                    num_workers=workers,
                    pin_memory=True,
                )
            )
    # Functions that take in predictions and labels and return a value
    metrics = {"cum_mse": cum_mse, "emd": emd_loss, "spearman": spearman_corr, "peak_rmse": peak_rmse}
    # Set up logging
    write_log = logs_path is not None
    if write_log:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if additional_name is not None:
            run_id = f"{run_id}_{additional_name}"
        path = logs_path / run_id
        writer = SummaryWriter(path)

        # Log model summary
        # shape = list(train_loader.dataset.data.shape)
        # shape[0] = batch_size
        # to_write = {"Model summary": get_summary_str(model, shape)}
        to_write = {}
        if additional_info:
            to_write.update(additional_info)

        for k, v in to_write.items():
            writer.add_text(k, v, global_step=0)

    # Set up optimizer and loss
    if use_class_weights:
        if class_weights is None:
            class_weights = calculate_class_weights(train_set, labels).to(DEVICE)
        else:
            class_weights = class_weights.to(DEVICE)
    else:
        class_weights = None
    model = model.to(DEVICE)
    if pretrain_fn is not None:
        loss = torch.nn.MSELoss()
    else:
        if whole_epoch or probabilistic_labels:
            # KLDivLoss for calculating loss between probability distributions
            # loss = torch.nn.CrossEntropyLoss()
            loss = torch.nn.KLDivLoss(reduction="none", log_target=False)
            # loss = SoftDTW(use_cuda=True, gamma=1.0)
            # loss = kl_div_loss_with_correlation_regularization
        else:
            # TODO: Think about ignore_index
            loss = torch.nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing, ignore_index=-1
            )
    opt = torch.optim.NAdam(model.parameters(), weight_decay=weight_decay, lr=lr)
    stopper = EarlyStopper()
    whole_epoch = whole_epoch or probabilistic_labels

    lowest_mean_val_loss = np.inf
    for epoch in range(epochs):
        with tqdm(total=len(train_loader), unit=" batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")

            # Train on batches in train_loader
            if pretrain_fn is None:
                batch_losses = train(
                    model,
                    train_loader,
                    opt,
                    loss,
                    progress=tepoch,
                    do_spectral_decoupling=do_spectral_decoupling,
                    whole_epoch=whole_epoch,
                    writer=writer,
                    epoch=epoch,
                    metrics=metrics,
                )
            else:
                batch_losses = pretrain_train(
                    model, train_loader, opt, loss, pretrain_fn, progress=tepoch
                )

            # Validate model and communicate results
            val_loss_list = []
            val_acc_list = []
            postfix_dict = {"loss": np.mean(batch_losses)}
            for i, val_loader in enumerate(val_loaders):
                if pretrain_fn is None:
                    val_losses, val_accuracy = validate(
                        model, val_loader, loss, whole_epoch, metrics, epoch, writer
                    )
                else:
                    val_losses = pretrain_validate(model, val_loader, loss, pretrain_fn)
                    val_accuracy = 0
                # Only count val_loss for first validation set
                if i == 0:
                    val_loss_list.append(val_losses)
                    val_acc_list.append(val_accuracy)
                postfix_dict[f"val_loss_{i}"] = np.mean(val_losses)
                postfix_dict[f"val_accuracy_{i}"] = val_accuracy
            tepoch.set_postfix(postfix_dict)
            mean_train_loss = np.mean(batch_losses)
            mean_val_loss = np.mean(val_loss_list[-1])

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
            if write_log:
                writer.add_scalar("train_loss", mean_train_loss, global_step=epoch)
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
    if pretrain_fn is None:
        if do_test_shuffled:
            results, _, _ = test_shuffled(
                model, test_loaders, loss, writer, labels, whole_epoch
            )
        else:
            results, _, _ = test(
                model, test_loaders, loss, writer, labels, whole_epoch, metrics, epoch
            )
    else:
        results = pretrain_test(model, test_loaders, loss, pretrain_fn, writer)
    return results


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.modules.loss._Loss,
    progress: tqdm = None,
    do_spectral_decoupling: bool = False,
    whole_epoch: bool = False,
    writer: SummaryWriter = None,
    epoch: int = None,
    metrics: dict = None,
) -> list[float]:
    """Train model for all batches, one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Loader that contains data used.
        optimizer (torch.optim.Optimizer): Optimizer used.
        loss_function (torch.nn.modules.loss._Loss): Loss function used.
        progress (tqdm, optional): tqdm instance to write progress to, will not write if not provided. Defaults to None.
        do_spectral_decoupling (bool, optional): Whether to apply spectral decoupling to loss. Defaults to False.

    Returns:
        list[float]: List containing loss for each batch.
    """
    model.train()
    if isinstance(loss_fn, torch.nn.KLDivLoss):
        softmax = torch.nn.LogSoftmax(dim=2)
    loss_per_batch = []
    dtw = SoftDTW(use_cuda=True, gamma=0.1, normalize=False)
    if metrics is not None:
        metric_values = defaultdict(lambda: [])
    for i, batch in enumerate(train_loader):
        # (Index, samples, channels), (Index, )
        data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

        optimizer.zero_grad()

        predictions = model(data)

        if labels.dim() > 1 and labels.shape[1] != predictions.shape[1]:
            labels = labels[:, : predictions.shape[1]]

        if len(predictions.shape) == 3:
            if whole_epoch and isinstance(loss_fn, torch.nn.KLDivLoss):
                loss, exp_loss, indiv_loss = kldiv_loss(
                    predictions.clone(), labels.clone(), dtw, mask=data == MASKING_VALUE
                )
                # Calculate training metrics
                if metrics is not None:
                    for metric_name, metric_fn in metrics.items():
                        metric_val = metric_fn(predictions.clone(), labels.clone())
                        metric_values[metric_name].append(metric_val)
                        # writer.add_scalar(
                        #     f"train_{metric_name}",
                        #     metric_val,
                        #     (epoch * progress.total) + progress.n,
                        # )
                # Individual loss metrics
                for metric_name, metric_batch_loss in indiv_loss.items():
                    metric_values[metric_name].append(metric_batch_loss.item())
                    # writer.add_scalar(
                    #     f"train_{metric_name}",
                    #     metric_batch_loss,
                    #     (epoch * progress.total) + progress.n,
                    # )
                # Class-wise loss
                for i_loss, loss_class in enumerate(exp_loss.mean(dim=[0, 1])):
                    writer.add_scalar(
                        f"train_loss_class{i_loss}",
                        loss_class,
                        (epoch * progress.total) + progress.n,
                    )

            elif loss_fn == kl_div_loss_with_correlation_regularization:
                loss = kl_div_loss_with_correlation_regularization(
                    predictions, labels, model, data
                )
            elif whole_epoch and isinstance(loss_fn, SoftDTW):
                predictions = torch.nn.functional.softmax(predictions, dim=2)
                predictions = predictions / predictions.sum()
                # loss = loss_fn(predictions, labels).mean()
                loss = []
                for l_i in range(labels.shape[-1]):
                    loss.append(
                        loss_fn(
                            predictions[..., l_i].unsqueeze(-1),
                            labels[..., l_i].unsqueeze(-1),
                        ).mean()
                    )
                loss = torch.stack(loss).mean()
            else:
                loss = loss_fn(
                    predictions.view(-1, predictions.shape[-1]), labels.flatten()
                )
        else:
            loss = loss_fn(predictions, labels)
        if do_spectral_decoupling:
            loss += 0.1 / 2 * (predictions**2).mean()

        loss_per_batch.append(loss.item())

        # Update loss shown every 5 batches, otherwise it is illegible
        if progress is not None:
            progress.update(1)
            if i % 5 == 0:
                progress.set_postfix(
                    {
                        "loss": round(np.mean(loss_per_batch), 5),
                    }
                )

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if metrics is not None:
        for metric_name, metric_batch_values in metric_values.items():
            writer.add_scalar(f"train_{metric_name}", np.mean(metric_batch_values), epoch)
    return loss_per_batch


def validate(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    whole_epoch: bool = False,
    metrics: dict = None,
    epoch: int = None,
    writer: SummaryWriter = None,
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

    if isinstance(loss_fn, torch.nn.KLDivLoss):
        softmax = torch.nn.LogSoftmax(dim=2)
    loss_per_batch = []
    total_correct = 0
    total_instances = 0
    dtw = SoftDTW(use_cuda=True, gamma=0.1, normalize=False)

    with torch.no_grad():
        all_labels = []
        all_preds = []
        if metrics is not None:
            metric_values = defaultdict(lambda: [])
        for batch_i, batch in enumerate(validation_loader):
            # (Index, samples, channels), (Index, )
            data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            predictions = model(data)

            dim = len(predictions.shape) - 1
            predicted_labels = torch.argmax(predictions, dim=dim)

            if labels.dim() > 1 and labels.shape[1] != predictions.shape[1]:
                labels = labels[:, : predictions.shape[1]]

            if not whole_epoch:
                matches = predicted_labels == labels
                correct_predictions = matches.sum().item()
            else:
                correct_predictions = labels.numel()

            total_correct += correct_predictions
            total_instances += labels.numel()
            # If data is sequence-shaped (batch, seq_len, class) instead of (batch, class)
            if metrics is not None:
                for metric_name, metric_fn in metrics.items():
                    metric_val = metric_fn(predictions.clone(), labels.clone())
                    metric_values[metric_name].append(metric_val)
                    # writer.add_scalar(
                    #     f"val_{metric_name}",
                    #     metric_val,
                    #     (epoch * len(validation_loader)) + batch_i,
                    # )
            if len(predictions.shape) == 3:
                if whole_epoch and isinstance(loss_fn, torch.nn.KLDivLoss):
                    # Add small value to prevent log(0), re-normalize
                    loss, _, _ = kldiv_loss(
                        predictions, labels, dtw, mask=data == MASKING_VALUE
                    )
                elif loss_fn == kl_div_loss_with_correlation_regularization:
                    loss = kl_div_loss_with_correlation_regularization(
                        predictions, labels, model, data
                    )
                elif whole_epoch and isinstance(loss_fn, SoftDTW):
                    loss = 0
                    for i in range(labels.shape[-1]):
                        loss += loss_fn(
                            predictions[..., i].unsqueeze(-1),
                            labels[..., i].unsqueeze(-1),
                        ).mean()
                    loss = loss / labels.shape[-1]
                else:
                    loss = loss_fn(
                        predictions.view(-1, predictions.shape[-1]), labels.flatten()
                    )
            else:
                loss = loss_fn(predictions, labels)
            all_labels.append(labels.cpu().flatten())
            all_preds.append(predicted_labels.cpu().flatten())
            loss_per_batch.append(loss.item())

        # Show test results for last batch
        if not whole_epoch:
            all_labels = torch.cat(all_labels, dim=0)
            all_preds = torch.cat(all_preds, dim=0)
            test_results = classification_report(
                all_labels,
                all_preds,
                output_dict=False,
                # target_names=class_labels,
                zero_division=0.0,
            )
            print(test_results)
    if metrics is not None:
        for metric_name, metric_batch_values in metric_values.items():
            writer.add_scalar(f"val_{metric_name}", np.mean(metric_batch_values), epoch)
    return loss_per_batch, round(total_correct / total_instances, 5)


def test(
    model: torch.nn.Module,
    test_loader: DataLoader | list[DataLoader],
    loss_fn: torch.nn.modules.loss._Loss,
    writer: SummaryWriter = None,
    class_labels: list[str] = None,
    whole_epoch: bool = False,
    metrics: dict = None,
    epoch: int = None,
) -> dict:
    """
    Test the PyTorch model on the given test data and return the classification report.

    Args:
        model (torch.nn.Module): The PyTorch model to test.
        test_loader (DataLoader): The DataLoader containing the test data.
        writer (SummaryWriter): The SummaryWriter to use for logging.

    Returns:
        dict: The classification report as a dictionary.
        torch.Tensor: The predicted classes.
        torch.Tensor: The true classes.
    """
    model.eval()
    dtw = SoftDTW(use_cuda=True, gamma=0.1, normalize=False)
    test_results = []

    if type(test_loader) is not list:
        # Assume type is DataLoader
        test_loader = [test_loader]
    for i, loader in enumerate(test_loader):
        outputs = []
        true_labels = torch.Tensor()
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                data, labels = batch[0].to(DEVICE), batch[1]
                predictions = model(data)
                # Cut off labels if needed
                if labels.dim() > 1 and labels.shape[1] != predictions.shape[1]:
                    labels = labels[:, : predictions.shape[1]]

                dim = len(predictions.shape) - 1
                if metrics is not None:
                    for metric_name, metric_fn in metrics.items():
                        metric_val = metric_fn(predictions.clone(), labels.clone())
                        writer.add_scalar(
                            f"test_{metric_name}",
                            metric_val,
                            (epoch * len(test_loader)) + batch_i,
                        )
                if not whole_epoch:
                    predictions = torch.argmax(predictions, dim=dim)
                    outputs = torch.cat([outputs, predictions.flatten().cpu()])
                    true_labels = torch.cat([true_labels, labels.flatten().cpu()])
                elif isinstance(loss_fn, torch.nn.KLDivLoss):
                    # Add small value to prevent log(0), re-normalize
                    loss, loss_raw, _ = kldiv_loss(
                        predictions, labels, dtw, mask=data == MASKING_VALUE
                    )
                    outputs.append(loss_raw.sum(dim=(1, 2)).to("cpu"))
                elif isinstance(loss_fn, SoftDTW):
                    loss = 0
                    labels = labels.to("cuda")
                    for i in range(labels.shape[-1]):
                        loss += loss_fn(
                            predictions[..., i].unsqueeze(-1),
                            labels[..., i].unsqueeze(-1),
                        ).mean()
                    loss = loss / labels.shape[-1]
                    outputs.append(loss.to("cpu"))

        if not whole_epoch:
            loader_results = classification_report(
                true_labels, outputs, output_dict=True, zero_division=0.0
            )
            test_results.append(loader_results)
            if writer is not None:
                writer.add_text(
                    f"Test results {i}", pretty_json(loader_results), global_step=0
                )
        else:
            outputs = torch.cat(outputs)
            loader_results = {"test_kldiv_list": outputs.tolist(), "test_kldiv_mean": torch.mean(outputs).item()}
            test_results.append(loader_results)

    return test_results, outputs, true_labels


def calculate_global_class_weights(
    datasets: list[Dataset],
    labels: list[str] = SAT1_STAGES_ACCURACY,
):
    counters = {(label, 0) for label in labels}
    for dataset in datasets:
        if dataset.split:
            # Split, count last dims in index_map
            labels = [idx[3] for idx in dataset.index_map]
            dataset_counter = Counter(labels)
            for key, value in dataset_counter.items():
                # Figure out solution, this is using dim indices instead of labels
                # can figure out labels by loading one sample maybe?
                # cant really load samples with specific labels
                pass
        else:
            pass
            # Not split, count occurrences of each label in data?


def calculate_class_weights(
    set: torch.utils.data.Dataset, labels: list[str]
) -> torch.Tensor:
    """
    Calculates class weights for a given dataset.

    Args:
        set (torch.utils.data.Dataset): The dataset to calculate class weights for.

    Returns:
        torch.Tensor: The calculated class weights.
    """
    occurrences = set.labels.unique(return_counts=True)
    weights = []
    for i in range(len(labels)):
        if i in occurrences[0]:
            weights.append(
                (
                    sum(occurrences[1])
                    / occurrences[1][(occurrences[0] == i).nonzero()[0]]
                ).item()
            )
        else:
            weights.append(1)
    return torch.Tensor(weights)


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
        elif np.isnan(validation_loss):
            return True
        return False


def kldiv_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    dtw: SoftDTW = None,
    mask: torch.Tensor = None,
):
    predictions = predictions.to(DEVICE)
    labels = labels.to(DEVICE)
    mask = ~mask[:, : predictions.shape[1], 0].unsqueeze(-1).to(DEVICE)

    # Predictions = model logits, non-softmaxed
    # labels = raw labels including negative, sums up to 1 at each time step

    predictions = torch.nn.functional.softmax(predictions, dim=2)

    # # FORWARD
    # Subset to 1024 since this dtw implementation does not support sequence length > 1024
    # dtw_loss = dtw(predictions[:, :1024, :], labels[:, :1024, :]).mean()

    # labels = labels + 1e-8
    # labels = labels / labels.sum(dim=-1, keepdim=True)

    forward_kl_loss = torch.nn.functional.kl_div(
        predictions.log(), labels, reduction="none"
    )

    # batchmean normalization
    # loss = forward_kl_loss.sum() / mask.sum()
    forward_kl_loss_norm = forward_kl_loss.sum() / predictions.shape[0]
    # 
    # mae_loss = torch.abs(predictions - labels).mean() * 1000
    # alpha = 0.7
    # loss = alpha * forward_kl_loss_norm + (1 - alpha) * mae_loss
    # loss = 0.5 * forward_kl_loss_norm + 0.5 * dtw_loss
    loss = forward_kl_loss_norm

    # Cross Entropy loss
    # loss = torch.nn.functional.cross_entropy(predictions[:,:,1:], labels[:,:,1:])
    return loss, forward_kl_loss, {"kldiv": forward_kl_loss_norm}