import torch
import numpy as np
from hmpai.utilities import MASKING_VALUE
from torch.utils.data import DataLoader, Dataset
from typing import Callable
from tqdm.notebook import tqdm
from hmpai.pytorch.utilities import (
    DEVICE,
    set_global_seed,
    save_model,
    load_model,
)
from torch.utils.tensorboard import SummaryWriter


def pretrain_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.modules.loss._Loss,
    pretrain_fn: Callable,
    progress: tqdm = None,
    scheduler: torch.optim.lr_scheduler = None,
):
    model.train()
    loss_per_batch = []
    for i, batch in enumerate(train_loader):
        # Changed to account for no labels in Dataset, maybe change back or make more complex if ever pretrain with Dataset that does include labels
        data = batch.to(DEVICE)
        # data, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
        # Generate pseudolabels in pretrain_fn
        data, labels = pretrain_fn(data)
        optimizer.zero_grad()

        predictions = model(data)

        # Cut off to end of trial
        if labels.shape[1] != predictions.shape[1]:
            labels = labels[:, : predictions.shape[1]]

        mask = torch.isnan(labels)
        loss = loss_fn(predictions[~mask], labels[~mask])

        loss_per_batch.append(loss.item())

        # Update loss shown every 5 batches, otherwise it is illegible
        if progress is not None:
            progress.update(1)
            if i % 5 == 0:
                progress.set_postfix({"loss": round(np.mean(loss_per_batch), 5)})

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
    return loss_per_batch


def pretrain_validate(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    pretrain_fn: Callable,
):
    model.eval()
    loss_per_batch = []

    with torch.no_grad():
        for batch in validation_loader:
            data = batch.to(DEVICE)
            data, labels = pretrain_fn(data)
            predictions = model(data)

            # Cut off to end of trial
            if labels.shape[1] != predictions.shape[1]:
                labels = labels[:, : predictions.shape[1]]

            mask = torch.isnan(labels)
            loss = loss_fn(predictions[~mask], labels[~mask])

            loss_per_batch.append(loss.item())

    return loss_per_batch


def pretrain_test(
    model: torch.nn.Module,
    test_loader: DataLoader | list[DataLoader],
    loss_fn: torch.nn.modules.loss._Loss,
    pretrain_fn: Callable,
    writer: SummaryWriter = None,
):
    model.eval()
    test_results = []
    if type(test_loader) is not list:
        # Assume type is DataLoader
        test_loader = [test_loader]
    for i, loader in enumerate(test_loader):
        loss_per_batch = []
        with torch.no_grad():
            for batch in loader:
                data = batch.to(DEVICE)
                data, labels = pretrain_fn(data)

                predictions = model(data)

                # Cut off to end of trial
                if labels.shape[1] != predictions.shape[1]:
                    labels = labels[:, : predictions.shape[1]]

                mask = torch.isnan(labels)
                loss = loss_fn(predictions[~mask], labels[~mask])

            loss_per_batch.append(loss.item())
        test_results.append(loss_per_batch)
        if writer is not None:
            writer.add_text(f"Test results {i}", str(loss_per_batch), global_step=0)
    return test_results

def random_masking(data: torch.Tensor, labels: torch.Tensor = None):
    batch_size, n_samples, n_channels = data.shape
    min_subsequence_length = 5
    mask_size = n_samples // min_subsequence_length
    pseudolabels = data.clone()


    mask = torch.randint(0, 2, (batch_size, mask_size), dtype=torch.float32, device=data.device)
    mask = mask.repeat_interleave(min_subsequence_length, dim=1)

    mask = mask.unsqueeze(2).expand(-1, -1, n_channels)

    data = mask * data

    return data, pseudolabels

# def random_masking(data: torch.Tensor, labels: torch.Tensor = None):
#     # Take input data (batch_size, seq_len, channels)
#     batch_size, n_samples, n_channels = data.shape
#     n_subsequences = 8
#     subseq_lengths = torch.tensor([11, 12, 13, 14, 15], device=data.device, dtype=torch.uint8)
#     pseudolabels = data.clone()

#     # Randomly choose subsequence lengths for each batch, channel, and subsequence
#     length_indices = torch.randint(0, len(subseq_lengths), (batch_size, n_channels, n_subsequences), device=data.device)
#     lengths = subseq_lengths[length_indices]

#     # Randomly choose start points for each subsequence
#     starts = torch.randint(0, n_samples, (batch_size, n_channels, n_subsequences), device=data.device)
    
#     # Calculate ends ensuring they don't go out of bounds
#     ends = torch.min(starts + lengths, torch.tensor(n_samples, device=data.device))

#     # Create a mask to apply the masking operation
#     mask = torch.ones((batch_size, n_samples, n_channels), device=data.device)

#     # Vectorized masking: Create ranges and apply the mask
#     for i in range(n_subsequences):
#         for j in range(batch_size):
#             for k in range(n_channels):
#                 mask[j, starts[j, k, i]:ends[j, k, i], k] = 0
    
#     # Apply the mask to the data
#     data = data * mask

#     return data, pseudolabels


# def random_masking(data: torch.Tensor, labels: torch.Tensor = None):
#     # Take input data (batch_size, seq_len, channels) and labels (batch_size, seq_len)
#     # Mask out random parts and create pseudolabels
#     # Pseudolabels is same as data, but with non-masked values set to NaN

#     mask = torch.full(data.shape, False, dtype=torch.bool)
#     labels = data.clone()
#     batch_size, seq_len, channels = data.shape
#     n_masks = 5
#     shortest_mask = 3
#     longest_mask = 30

#     for i in range(batch_size):
#         rt_idx = torch.eq(data[i, :, 0], MASKING_VALUE).nonzero()
#         if rt_idx.numel() > 0:
#             rt_idx = rt_idx[0].item()
#         else:
#             rt_idx = seq_len
#         starts = np.random.randint(0, rt_idx, size=n_masks)
#         lengths = np.random.randint(shortest_mask, longest_mask, size=n_masks)

#         for start, length in zip(starts, lengths):
#             # Find first index of MASKING_VALUE
#             if start + length > rt_idx:
#                 length = rt_idx - start
#             mask[i, start : start + length, :] = True

#         data[mask] = 0
#         labels[~mask] = np.NaN

#     return data, labels

# def random_masking(data: torch.Tensor, labels: torch.Tensor = None):
#     # Take input data (batch_size, seq_len, channels) and labels (batch_size, seq_len)
#     # Mask out random parts and create pseudolabels
#     # Pseudolabels is same as data, but with non-masked values set to NaN

#     batch_size, seq_len, channels = data.shape
#     n_masks = 5
#     shortest_mask = 3
#     longest_mask = 30

#     # Create mask tensor
#     mask = torch.zeros(data.shape, dtype=torch.bool, device=DEVICE)

#     # Generate random starts and lengths for each batch
#     rt_idx = (data[:, :, 0] == MASKING_VALUE).int().argmax(dim=1)
#     rt_idx = torch.where(rt_idx == 0, torch.tensor(1, dtype=rt_idx.dtype), rt_idx)
#     starts = [torch.randint(0, rt_idx[i].item(), (n_masks,), device=DEVICE) for i in range(batch_size)]
#     lengths = torch.randint(shortest_mask, longest_mask, (batch_size, n_masks), device=DEVICE)

#     for i in range(batch_size):
#         for start, length in zip(starts[i], lengths[i]):
#             if start + length > rt_idx[i]:
#                 length = rt_idx[i] - start
#             mask[i, start:start + length, :] = True

#     data[mask] = 0
#     pseudolabels = data.clone()
#     pseudolabels[~mask] = float('nan')

#     return data, pseudolabels