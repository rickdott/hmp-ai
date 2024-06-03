import torch
import numpy as np
from hmpai.utilities import MASKING_VALUE

def random_masking(data: torch.Tensor, labels: torch.Tensor = None):
    # Take input data (batch_size, seq_len, channels) and labels (batch_size, seq_len)
    # Mask out random parts and create pseudolabels
    # Pseudolabels is same as data, but with non-masked values set to NaN

    mask = torch.full(data.shape, False, dtype=torch.bool)
    labels = data.clone()
    batch_size, seq_len, channels = data.shape
    n_masks = 5
    shortest_mask = 3
    longest_mask = 30

    for i in range(batch_size):
        # TODO: Change so seq_len is rt idx? Loss is nan when nothing is masked
        rt_idx = torch.eq(data[i,:,0], MASKING_VALUE).nonzero()
        if rt_idx.numel() > 0:
            rt_idx = rt_idx[0].item()
        else:
            rt_idx = seq_len
        starts = np.random.randint(0, rt_idx, size=n_masks)
        lengths = np.random.randint(shortest_mask, longest_mask, size=n_masks)

        for start, length in zip(starts, lengths):
            # Find first index of MASKING_VALUE
            if start + length > rt_idx:
                length = rt_idx - start
            mask[i, start:start+length, :] = True

        data[mask] = 0
        labels[~mask] = np.NaN
        
    return data, labels