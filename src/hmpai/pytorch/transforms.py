import torch
from hmpai.utilities import MASKING_VALUE
import random
import numpy as np

class ShuffleOperations(object):
    def __init__(self):
        pass

    def __call__(self, data):
        labels = data[1]
        data = data[0]
        length = data.shape[0]
        rt_idx = (data[:, 0] != MASKING_VALUE).nonzero(as_tuple=True)[0][-1].item() + 1

        current_label = labels[0]
        sequence_indices = []
        start_idx = 0

        for i in range(1, rt_idx):
            if labels[i] != current_label:
                sequence_indices.append((start_idx, i, current_label))
                start_idx = i
                current_label = labels[i]

        sequence_indices = torch.tensor(sequence_indices)

        if len(sequence_indices) > 0:
            perm = torch.randperm(len(sequence_indices))
            sequence_indices = sequence_indices[perm]
            shuffled_data = torch.full_like(data, fill_value=MASKING_VALUE, dtype=data.dtype)
            shuffled_labels = torch.full_like(labels, fill_value=0, dtype=labels.dtype)
            
            current_idx = 0

            for seq in sequence_indices:
                start_idx, end_idx, label = seq
                start_idx = start_idx.item()
                end_idx = end_idx.item()
                length = end_idx - start_idx
                shuffled_data[current_idx:current_idx + length] = data[start_idx:end_idx]
                shuffled_labels[current_idx:current_idx + length] = labels[start_idx:end_idx]
                current_idx += length
            
            return shuffled_data, shuffled_labels
        else:
            return data, labels





class RandomCropTransform(object):
    def __init__(self):
        pass


    def __call__(self, data):
        labels = data[1]
        data = data[0]
        length = data.shape[0]

        rt_idx = (data[:, 0] != MASKING_VALUE).nonzero(as_tuple=True)[0][-1].item() + 1
        # Crop can only be from 10% after start
        crop_length = random.randint(rt_idx // 10, rt_idx)
        start = random.randint(0, max(0, rt_idx - crop_length))
        end = start + crop_length
        cropped_data = data[start:end, :]
        cropped_labels = labels[start:end]

        padded_data = torch.full((length, cropped_data.shape[1]), MASKING_VALUE, dtype=cropped_data.dtype)
        padded_labels = torch.full((length, ), 0, dtype=cropped_labels.dtype)

        padded_data[:cropped_data.shape[0],:] = cropped_data
        padded_labels[:cropped_labels.shape[0]] = cropped_labels

        return padded_data, padded_labels


class EegNoiseTransform(object):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, segment):
        return segment


class EegTimeWarpingTransform(object):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, segment):
        return segment


class EegChannelShufflingTransform(object):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, segment):
        return segment
