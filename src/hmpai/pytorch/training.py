from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
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
    max_channels = max(x.shape[1] for x in xs)  # Subtract 1 to account for positional encoding channel
    max_label_len = max(y.shape[0] for y in ys)
    max_label_width = max(y.shape[1] for y in ys)

    max_coord_channels = max_channels - 1
    
    # Pad sequences to maximum length
    padded_xs = []
    padded_ys = []
    
    for x, y in zip(xs, ys):
        seq_len = x.shape[0]
        ch_len = x.shape[1]
        label_len = y.shape[0]
        label_width = y.shape[1]
        # Pad data tensor
        if seq_len < max_seq_len:
            pad_size = max_seq_len - seq_len
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, pad_size), value=MASKING_VALUE)
        else:
            x_padded = x
        if ch_len < max_channels:
            ch_pad_size = max_channels - ch_len
            # Front pad so that pos enc stays last index
            x_padded = torch.nn.functional.pad(x_padded, (ch_pad_size, 0, 0, 0), value=MASKING_VALUE)
        padded_xs.append(x_padded)
        
        # Pad label tensor
        if label_len < max_label_len:
            pad_size = max_label_len - label_len
            y_padded = torch.nn.functional.pad(y, (0, 0, 0, pad_size), value=0.0)
        else:
            y_padded = y
        if label_width < max_label_width:
            width_pad_size = max_label_width - label_width
            y_padded = torch.nn.functional.pad(y_padded, (0, width_pad_size, 0, 0), value=MASKING_VALUE)
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
                if key == 'coords':
                    if value.shape[0] < max_coord_channels:
                        value = torch.nn.functional.pad(value, (0, 0, max_coord_channels - value.shape[0], 0), value=MASKING_VALUE) 

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
    torch.set_float32_matmul_precision("high")

    # Create loaders
    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=mixed_collate
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
                )
            )
    # Set up logging
    write_log = logs_path is not None
    writer = None
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
    opt = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=lr, fused=True)
    # scaler = torch.amp.GradScaler('cuda')
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
                # scaler=scaler,
            )

            # Validate model and communicate results
            val_loss_list = []
            val_acc_list = []
            epoch_val_loss = 0
            postfix_dict = {"loss": np.mean(batch_losses)}
            for i, val_loader in enumerate(val_loaders):
                val_losses = validate(model, val_loader, loss)

                epoch_val_loss += np.mean(val_losses)
                postfix_dict[f"val_loss_{i}"] = np.mean(val_losses)
            mean_train_loss = np.mean(batch_losses)
            mean_val_loss = epoch_val_loss / len(val_loaders)
            postfix_dict["mean_val_loss"] = mean_val_loss
            tepoch.set_postfix(postfix_dict)


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
        # print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")
    # Re-load best performing model
    if write_log:
        best_checkpoint = load_model(path / "checkpoint.pt")
        model.load_state_dict(best_checkpoint["model_state_dict"])
        opt.load_state_dict(best_checkpoint["optimizer_state_dict"])
        epoch = best_checkpoint["epoch"]
        loss = best_checkpoint["loss"]

    # Test model
    if len(test_loaders) > 0:
        results = test(model, test_loaders, loss)
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
    scaler = None,
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
    amp_dtype = torch.bfloat16

    loss_per_batch = []
    for i, batch in enumerate(train_loader):
        # (Index, samples, channels), (Index, )
        data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
        info = batch[2] if len(batch) > 2 else None
        padding_mask = batch[3].to(DEVICE) if len(batch) > 3 else None

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            if 'coords' in info:
                predictions = model(data, task=info["task"] if info is not None else None, coords=info["coords"].to(DEVICE))
            else:
                predictions = model(data, task=info["task"] if info is not None else None)

            if labels.dim() > 1 and labels.shape[1] != predictions.shape[1]:
                labels = labels[:, : predictions.shape[1]]
                if padding_mask is not None:
                    padding_mask = padding_mask[:, : predictions.shape[1]]

            loss, exp_loss, indiv_loss = loss_fn(predictions, labels, padding_mask)

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
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # scheduler.step()
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

            if 'coords' in info:
                predictions = model(data, task=info["task"] if info is not None else None, coords=info["coords"].to(DEVICE))
            else:
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
        loss_per_batch = []
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                info = batch[2] if len(batch) > 2 else None
                padding_mask = batch[3].to(DEVICE) if len(batch) > 3 else None
                if 'coords' in info:
                    predictions = model(data, task=info["task"] if info is not None else None, coords=info["coords"].to(DEVICE))
                else:
                    predictions = model(data, task=info["task"] if info is not None else None)
                # Cut off labels if needed
                if labels.dim() > 1 and labels.shape[1] != predictions.shape[1]:
                    labels = labels[:, : predictions.shape[1]]
                    if padding_mask is not None:
                        padding_mask = padding_mask[:, : predictions.shape[1]]

                loss, _, _ = loss_fn(predictions, labels, padding_mask)
                loss_per_batch.append(loss)


        loss_per_batch = torch.stack(loss_per_batch)
        loader_results = {
            "test_kldiv_list": loss_per_batch.tolist(),
            "test_kldiv_mean": torch.mean(loss_per_batch).item(),
        }
        test_results.append(loader_results)

    return test_results


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
            along the last dimension and can include negative values. Positions with
            MASKING_VALUE are considered invalid/padded classes.
        padding_mask (torch.Tensor, optional): Boolean mask with shape (batch_size, sequence_length)
            where True indicates valid positions and False indicates padding. If None, no masking is applied.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            - loss (torch.Tensor): The normalized forward KL divergence loss.
            - forward_kl_loss (torch.Tensor): The element-wise KL divergence loss 
              before normalization (with invalid positions zeroed out).
            - metrics (Dict[str, torch.Tensor]): A dictionary containing the normalized 
              KL divergence loss under the key "kldiv".

    Notes:
        - The predictions are softmaxed along the last dimension before computing the 
          KL divergence.
        - When padding_mask is provided, loss is only computed on valid (non-padded) positions.
        - Classes with MASKING_VALUE in labels are excluded from loss calculation.
        - The loss is normalized by the number of valid elements (time + class dimensions).
    """

    # Create mask for valid classes (not padded with MASKING_VALUE)
    # Shape: [B, T, C]
    class_mask = (labels != MASKING_VALUE)
    
    predictions_masked = predictions.clone()
    predictions_masked = torch.where(class_mask, predictions_masked, torch.tensor(float('-inf'), device=predictions.device))
    
    # Now softmax only normalizes over valid classes
    predictions = torch.nn.functional.softmax(predictions_masked, dim=2)
    predictions = torch.clamp(predictions, 1e-8, 1.0)
    
    # Replace MASKING_VALUE in labels with 0 to avoid numerical issues in KL div
    labels_clean = torch.where(class_mask, labels, torch.zeros_like(labels))
    
    forward_kl_loss = torch.nn.functional.kl_div(
        predictions.log(), labels_clean, reduction="none"
    )

    # Apply masking
    if padding_mask is not None:
        # Combine time-based padding mask with class-based mask
        # padding_mask: [B, T] -> [B, T, 1]
        # class_mask: [B, T, C]
        mask_expanded = padding_mask.unsqueeze(-1).expand_as(forward_kl_loss)
        combined_mask = mask_expanded & class_mask
        
        # Zero out loss at invalid positions
        forward_kl_loss = forward_kl_loss * combined_mask.float()
        # Normalize by number of valid positions
        num_valid = combined_mask.sum()
        forward_kl_loss_norm = forward_kl_loss.sum() / num_valid if num_valid > 0 else forward_kl_loss.sum()
    else:
        # Only apply class mask
        forward_kl_loss = forward_kl_loss * class_mask.float()
        num_valid = class_mask.sum()
        forward_kl_loss_norm = forward_kl_loss.sum() / num_valid if num_valid > 0 else forward_kl_loss.sum()

    loss = forward_kl_loss_norm

    return loss, forward_kl_loss, {"kldiv": forward_kl_loss_norm}

