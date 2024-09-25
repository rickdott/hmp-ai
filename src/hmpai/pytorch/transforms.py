import torch
from hmpai.utilities import MASKING_VALUE
import random
import numpy as np
import braindecode.augmentation as aug


class GaussianNoise(object):
    def __init__(self, probability=0.5, std=0.16):
        self.augment = aug.GaussianNoise(probability, std)

    def __call__(self, data):
        labels = data[1]
        data = data[0]
        data = data.transpose(1, 0)
        data = self.augment(data)
        data = data.transpose(1, 0)
        return data, labels


class TimeReverse(object):
    def __init__(self, probability=0.5):
        self.augment = aug.TimeReverse(probability)

    def __call__(self, data):
        labels = data[1]
        data = data[0]
        data = data.transpose(1, 0)
        data = self.augment(data)
        data = data.transpose(1, 0)
        return data, labels


# Inspired by smooth_time_mask from braindecode
class TimeMasking(object):
    def __init__(self, probability=0.5, mask_len_samples=20):
        self.mask_len_samples = mask_len_samples
        self.probability = probability

    def __call__(self, data):
        labels = data[1]
        data = data[0]
        length, channels = data.shape

        if random.random() > self.probability:
            return data, labels
        
        rt_idx = (data[:, 0] != MASKING_VALUE).nonzero(as_tuple=True)[0][-1].item() + 1
        max_length = rt_idx - self.mask_len_samples
        # TODO: Fix when max_length < 0 (short stages in segment pred)
        start = random.randint(0, max_length)

        t = torch.arange(length, device=data.device).float()
        t = t.repeat(channels, 1)

        s = 1000 / length
        mask = (torch.sigmoid(s * -(t - start)) +
            torch.sigmoid(s * (t - start - self.mask_len_samples))
            ).float().to(data.device)
        data = data * mask.transpose(1, 0)
        labels[start:start+self.mask_len_samples] = 0
        # data = data.transpose(1, 0).unsqueeze(0)
        # data, labels = self.augment(data, labels, torch.Tensor(start), self.mask_len_samples)
        # data = data.squeeze().transpose(1, 0)

        return data, labels


class ChannelsDropout(object):
    def __init__(self, probability=0.5, p_drop=0.2):
        self.augment = aug.ChannelsDropout(probability, p_drop)

    def __call__(self, data):
        labels = data[1]
        data = data[0]
        data = data.transpose(1, 0)
        data = self.augment(data)
        data = data.transpose(1, 0)
        return data, labels


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
            shuffled_data = torch.full_like(
                data, fill_value=MASKING_VALUE, dtype=data.dtype
            )
            shuffled_labels = torch.full_like(labels, fill_value=0, dtype=labels.dtype)

            current_idx = 0

            for seq in sequence_indices:
                start_idx, end_idx, label = seq
                start_idx = start_idx.item()
                end_idx = end_idx.item()
                length = end_idx - start_idx
                shuffled_data[current_idx : current_idx + length] = data[
                    start_idx:end_idx
                ]
                shuffled_labels[current_idx : current_idx + length] = labels[
                    start_idx:end_idx
                ]
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
        start = random.randint(0, rt_idx)
        crop_length = random.randint(5, 15)
        end = start + crop_length
        cropped_data = data[start:end, :]
        cropped_labels = labels[start:end]

        padded_data = torch.full(
            (length, cropped_data.shape[1]), MASKING_VALUE, dtype=cropped_data.dtype
        )
        padded_labels = torch.full((length,), 0, dtype=cropped_labels.dtype)

        padded_data[: cropped_data.shape[0], :] = cropped_data
        padded_labels[: cropped_labels.shape[0]] = cropped_labels

        return padded_data, padded_labels
    
class FixedLengthCropTransform(object):
    CROP_LENGTH = 50

    def __init__(self):
        pass

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        end_idx = (data[:, 0] != MASKING_VALUE).nonzero(as_tuple=True)[0][-1].item() + 1
        start = torch.randint(0, max(end_idx - self.CROP_LENGTH, 1), (1,))
        end = start + self.CROP_LENGTH
        # print(start, end)
        cropped_data = data[start:end, :]
        cropped_labels = labels[start:end, :]

        if cropped_data.shape[0] < self.CROP_LENGTH:
            offset = self.CROP_LENGTH - cropped_data.shape[0]
            cropped_data = torch.nn.functional.pad(cropped_data, (0, 0, 0, offset), value=MASKING_VALUE)
            cropped_labels = torch.nn.functional.pad(cropped_labels, (0, 0, 0, offset), value=0)
            cropped_labels[-offset:,0] = 1.0

        return cropped_data, cropped_labels

class StartJitterTransform(object):
    def __init__(self, offset_before):
        self.offset_before = offset_before
    
    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        offset = torch.randint(self.offset_before, (1,))

        cropped_data = data[offset:, :]
        cropped_labels = labels[offset:, :]

        cropped_data = torch.nn.functional.pad(cropped_data, (0, 0, 0, offset), value=MASKING_VALUE)
        cropped_labels = torch.nn.functional.pad(cropped_labels, (0, 0, 0, offset), value=0)
        cropped_labels[-offset:,0] = 1.0

        return cropped_data, cropped_labels



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
