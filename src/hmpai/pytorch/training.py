from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm
from hmpai.utilities import MASKING_VALUE
from hmpai.pytorch.utilities import (
    DEVICE,
    set_global_seed,
    save_model,
    load_model,
)
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

def mixed_collate(batch):
    """
    Custom collate function to handle variable-length sequences with padding and combine metadata.
    Pads sequences to the maximum length in the batch.
    """
    xs, ys, meta = [], [], []

    # Extract tensors and metadata
    for sample_data, sample_label, sample_info in batch:
        xs.append(sample_data)
        ys.append(sample_label)
        meta.append(sample_info)  # List of dicts containing metadata
    
    # Find maximum sequence length in the batch
    max_seq_len = max(x.shape[0] for x in xs)
    max_label_len = max(y.shape[0] for y in ys)
    
    # Pad sequences to maximum length
    padded_xs = []
    padded_ys = []
    
    for x, y in zip(xs, ys):
        seq_len = x.shape[0]
        label_len = y.shape[0]
        
        # Pad data tensor
        if seq_len < max_seq_len:
            pad_size = max_seq_len - seq_len
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, pad_size), value=MASKING_VALUE)
        else:
            x_padded = x
        padded_xs.append(x_padded)
        
        # Pad label tensor
        if label_len < max_label_len:
            pad_size = max_label_len - label_len
            y_padded = torch.nn.functional.pad(y, (0, 0, 0, pad_size), value=0.0)
        else:
            y_padded = y
        padded_ys.append(y_padded)
    
    # Stack padded tensors
    xs = torch.stack(padded_xs)  # [B, T, C] (EEG data)
    ys = torch.stack(padded_ys)  # [B, T, num_classes] (labels)
    
    # Create padding mask (True for valid data, False for padding)
    # Check if any channel contains MASKING_VALUE to determine padding
    padding_mask = (xs != MASKING_VALUE).any(dim=-1)  # [B, T]

    # Combine metadata into a single dictionary
    combined_meta = {}
    
    # Iterate through metadata and combine
    for sample_info in meta:
        for key, value in sample_info[0].items():
            if key not in combined_meta:
                combined_meta[key] = []

            # If the value is a tensor (e.g., epoch, cluster, channel positions), stack them
            if isinstance(value, torch.Tensor):
                combined_meta[key].append(value)
            else:
                # Otherwise, just append the values (lists of strings, like task, participant)
                combined_meta[key].append(value)

    # Convert lists of metadata values into tensors if needed
    for key, value in combined_meta.items():
        if isinstance(value[0], torch.Tensor):
            combined_meta[key] = torch.stack(value)  # Convert list of tensors into a tensor
        else:
            combined_meta[key] = value  # Keep list if it's not a tensor

    return xs, ys, combined_meta, padding_mask


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
    weight_decay: float = 0.0,
    lr: float = 0.002,  # Default learning rate for optimizer
    seed: int = 42,
) -> dict:
    """
    Trains and evaluates a PyTorch model using the provided datasets.

    Args:
        model (torch.nn.Module): The PyTorch model to train and evaluate.
        train_set (Dataset): The dataset used for training.
        test_set (Dataset | list[Dataset]): The dataset(s) used for testing. Can be a single Dataset or a list of Datasets.
        val_set (Dataset, optional): The dataset used for validation. Can be a single Dataset or a list of Datasets. Defaults to None.
        batch_size (int, optional): The batch size for data loaders. Defaults to 128.
        epochs (int, optional): The number of training epochs. Defaults to 20.
        workers (int, optional): The number of worker threads for data loading. Defaults to 4.
        logs_path (Path, optional): The directory path to save training logs and checkpoints. If None, logging is disabled. Defaults to None.
        additional_info (dict, optional): Additional information to log as text. Defaults to None.
        additional_name (str, optional): Additional name to append to the log directory. Defaults to None.
        weight_decay (float, optional): Weight decay (L2 regularization) for the optimizer. Defaults to 0.0.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.002.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        dict: A dictionary containing the test results.

    Notes:
        - The function uses early stopping to terminate training if validation loss does not improve sufficiently.
        - The best-performing model (based on validation loss) is saved and reloaded for testing.
        - If `logs_path` is provided, training logs and model checkpoints are saved to the specified directory.
    """
    set_global_seed(seed)
    torch.cuda.empty_cache()
    # Create loaders with optimizations
    train_loader = DataLoader(
        train_set, 
        batch_size, 
        shuffle=True, 
        num_workers=workers, 
        pin_memory=True, 
        collate_fn=mixed_collate,
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=8 if workers > 0 else None,
    )
    # Do not shuffle test loader since testing should be the same always
    test_loaders = []
    if isinstance(test_set, list):
        for test_data in test_set:
            test_loaders.append(
                DataLoader(
                    test_data,
                    batch_size,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                    collate_fn=mixed_collate,
                    # persistent_workers=True if workers > 0 else False,
                    # prefetch_factor=2 if workers > 0 else None,
                )
            )
    elif isinstance(test_set, Dataset):
        # Assume type of test_set is Dataset
        test_loaders.append(
            DataLoader(
                test_set,
                batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
                collate_fn=mixed_collate,
                # persistent_workers=True if workers > 0 else False,
                # prefetch_factor=2 if workers > 0 else None,
            )
        )

    val_loaders = []
    if val_set is not None:
        if isinstance(val_set, list):
            for val in val_set:
                val_loaders.append(
                    DataLoader(
                        val,
                        batch_size,
                        shuffle=False,
                        num_workers=workers,
                        pin_memory=True,
                        collate_fn=mixed_collate,
                        # persistent_workers=True if workers > 0 else False,
                        # prefetch_factor=2 if workers > 0 else None,
                    )
                )
        elif isinstance(val_set, Dataset):
            val_loaders.append(
                DataLoader(
                    val_set,
                    batch_size,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                    collate_fn=mixed_collate,
                    # persistent_workers=True if workers > 0 else False,
                    # prefetch_factor=2 if workers > 0 else None,
                )
            )
    # Set up logging
    write_log = logs_path is not None
    if write_log:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if additional_name is not None:
            run_id = f"{run_id}_{additional_name}"
        path = logs_path / run_id
        writer = SummaryWriter(path)

        to_write = {}
        if additional_info:
            to_write.update(additional_info)

        for k, v in to_write.items():
            writer.add_text(k, v, global_step=0)

    model = model.to(DEVICE)

    loss = kldiv_loss

    # opt = torch.optim.NAdam(model.parameters(), weight_decay=weight_decay, lr=lr)
    opt = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    stopper = EarlyStopper(tolerance=5)

    lowest_mean_val_loss = np.inf
    for epoch in range(epochs):
        with tqdm(total=len(train_loader), unit=" batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")

            # Train on batches in train_loader
            batch_losses = train(
                model,
                train_loader,
                opt,
                scheduler,
                loss,
                progress=tepoch,
                writer=writer,
                epoch=epoch,
            )


            # Validate model and communicate results
            val_loss_list = []
            postfix_dict = {"loss": np.mean(batch_losses)}
            for i, val_loader in enumerate(val_loaders):
                val_losses = validate(model, val_loader, loss)

                # Only count val_loss for first validation set
                if i == 0:
                    val_loss_list.append(val_losses)
                postfix_dict[f"val_loss_{i}"] = np.mean(val_losses)
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
                writer.flush()

            # Stop training if validation loss has not improved sufficiently
            if stopper.check_stop(mean_val_loss):
                break
        print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")
    # Re-load best performing model
    if write_log:
        best_checkpoint = load_model(path / "checkpoint.pt")
        model.load_state_dict(best_checkpoint["model_state_dict"])
        opt.load_state_dict(best_checkpoint["optimizer_state_dict"])
        epoch = best_checkpoint["epoch"]
        loss = best_checkpoint["loss"]

    # Test model
    if len(test_loaders) > 0:
        results, _, = test(model, test_loaders, loss)
    else:
        results = None
    return results


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: torch.nn.modules.loss._Loss,
    progress: tqdm = None,
    writer: SummaryWriter = None,
    epoch: int = None,
) -> list[float]:
    """
    Trains a PyTorch model for one epoch using the provided data loader, optimizer, and loss function.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        loss_fn (torch.nn.modules.loss._Loss): Loss function to compute the training loss.
        progress (tqdm, optional): tqdm progress bar instance for tracking training progress. Defaults to None.
        writer (SummaryWriter, optional): TensorBoard SummaryWriter for logging training metrics. Defaults to None.
        epoch (int, optional): Current epoch number, used for logging. Defaults to None.

    Returns:
        list[float]: A list of loss values for each batch in the training epoch.
    """
    model.train()

    loss_per_batch = []
    
    for i, batch in enumerate(train_loader):
        # (Index, samples, channels), (Index, )
        data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
        info = batch[2] if len(batch) > 2 else None
        padding_mask = batch[3].to(DEVICE) if len(batch) > 3 else None
        optimizer.zero_grad()

        # Mixed precision forward pass
        predictions = model(data, task=info["task"] if info is not None else None)

        if labels.dim() > 1 and labels.shape[1] != predictions.shape[1]:
            labels = labels[:, : predictions.shape[1]]
            if padding_mask is not None:
                padding_mask = padding_mask[:, : predictions.shape[1]]

        loss, exp_loss, indiv_loss = loss_fn(predictions.clone(), labels.clone(), padding_mask)
        for i_loss, loss_class in enumerate(exp_loss.mean(dim=[0, 1])):
            writer.add_scalar(
                f"train_loss_class{i_loss}",
                loss_class,
                (epoch * progress.total) + progress.n,
            )

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
        optimizer.step()
        scheduler.step()
    
    return loss_per_batch


def validate(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
) -> list[float]:
    """
    Validate the performance of a model on a validation dataset.

    This function evaluates the model in evaluation mode using the provided
    validation data loader and computes the loss for each batch. It ensures
    that the model's gradients are not updated during validation by using
    `torch.no_grad()`.

        model (torch.nn.Module): The PyTorch model to validate.
        validation_loader (DataLoader): DataLoader providing the validation dataset.
        loss_fn (torch.nn.modules.loss._Loss): The loss function used to compute the loss.

        list[float]: A list containing the loss value for each batch in the validation dataset.
    """
    model.eval()

    loss_per_batch = []

    with torch.no_grad():
        for batch_i, batch in enumerate(validation_loader):
            # (Index, samples, channels), (Index, )
            data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            info = batch[2] if len(batch) > 2 else None
            padding_mask = batch[3].to(DEVICE) if len(batch) > 3 else None

            predictions = model(data, task=info["task"] if info is not None else None)

            if labels.dim() > 1 and labels.shape[1] != predictions.shape[1]:
                labels = labels[:, : predictions.shape[1]]
                if padding_mask is not None:
                    padding_mask = padding_mask[:, : predictions.shape[1]]

            loss, _, _ = loss_fn(predictions, labels, padding_mask)
            loss_per_batch.append(loss.item())

    return loss_per_batch


def test(
    model: torch.nn.Module,
    test_loader: DataLoader | list[DataLoader],
    loss_fn: torch.nn.modules.loss._Loss,
) -> dict:
    """
    Evaluate a PyTorch model on one or more test datasets using a specified loss function.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        test_loader (DataLoader | list[DataLoader]): A single DataLoader or a list of DataLoaders
            containing the test datasets.
        loss_fn (torch.nn.modules.loss._Loss): The loss function to compute the evaluation metric.

    Returns:
        tuple: A tuple containing:
            - test_results (list[dict]): A list of dictionaries, one for each DataLoader, containing:
                - "test_kldiv_list" (list[float]): A list of per-sample loss values.
                - "test_kldiv_mean" (float): The mean loss value across all samples in the DataLoader.
            - outputs (torch.Tensor): A tensor containing the concatenated per-sample loss values
              across all DataLoaders.
    """
    model.eval()
    test_results = []

    if type(test_loader) is not list:
        # Assume type is DataLoader
        test_loader = [test_loader]
    for i, loader in enumerate(test_loader):
        outputs = []
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                info = batch[2] if len(batch) > 2 else None
                padding_mask = batch[3].to(DEVICE) if len(batch) > 3 else None
                predictions = model(data, task=info["task"] if info is not None else None)
                # Cut off labels if needed
                if labels.dim() > 1 and labels.shape[1] != predictions.shape[1]:
                    labels = labels[:, : predictions.shape[1]]
                    if padding_mask is not None:
                        padding_mask = padding_mask[:, : predictions.shape[1]]

                loss, loss_raw, _ = loss_fn(predictions, labels, padding_mask)
                outputs.append(loss_raw.sum(dim=(1, 2)).to("cpu"))

        outputs = torch.cat(outputs)
        loader_results = {
            "test_kldiv_list": outputs.tolist(),
            "test_kldiv_mean": torch.mean(outputs).item(),
        }
        test_results.append(loader_results)

    return test_results, outputs


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
    padding_mask: torch.Tensor = None,
):
    """
    Computes the Kullback-Leibler divergence (KLDiv) loss between predictions and labels.

    Args:
        predictions (torch.Tensor): The model logits (non-softmaxed) with shape 
            (batch_size, sequence_length, num_classes).
        labels (torch.Tensor): The target labels with shape 
            (batch_size, sequence_length, num_classes). The labels should sum up to 1 
            along the last dimension and can include negative values.
        padding_mask (torch.Tensor, optional): Boolean mask with shape (batch_size, sequence_length)
            where True indicates valid positions and False indicates padding. If None, no masking is applied.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            - loss (torch.Tensor): The normalized forward KL divergence loss.
            - forward_kl_loss (torch.Tensor): The element-wise KL divergence loss 
              before normalization.
            - metrics (Dict[str, torch.Tensor]): A dictionary containing the normalized 
              KL divergence loss under the key "kldiv".

    Notes:
        - The predictions are softmaxed along the last dimension before computing the 
          KL divergence.
        - When padding_mask is provided, loss is only computed on valid (non-padded) positions.
        - The loss is normalized by the number of valid positions (or batch size if no mask).
    """
    predictions = predictions.to(DEVICE)
    labels = labels.to(DEVICE)

    # Predictions = model logits, non-softmaxed
    # labels = raw labels including negative, sums up to 1 at each time step
    predictions = torch.nn.functional.softmax(predictions, dim=2)
    predictions = torch.clamp(predictions, min=1e-7, max=1.0)  # Prevent log(0)

    forward_kl_loss = torch.nn.functional.kl_div(
        predictions.log(), labels, reduction="none"
    )

    # Apply masking if provided
    if padding_mask is not None:
        # Expand mask to match loss dimensions [B, T, C]
        mask_expanded = padding_mask.unsqueeze(-1).expand_as(forward_kl_loss)
        # Zero out loss at padded positions
        forward_kl_loss = forward_kl_loss * mask_expanded.float()
        # Normalize by number of valid positions
        num_valid = padding_mask.sum()
        forward_kl_loss_norm = forward_kl_loss.sum() / num_valid if num_valid > 0 else forward_kl_loss.sum()
    else:
        # Original batchmean normalization
        forward_kl_loss_norm = forward_kl_loss.sum() / predictions.shape[0]

    loss = forward_kl_loss_norm

    return loss, forward_kl_loss, {"kldiv": forward_kl_loss_norm}

