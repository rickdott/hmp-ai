import torch
from hmpai.utilities import MASKING_VALUE, get_masking_indices, get_masking_index
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

        if torch.rand((1,)).item() < self.probability:
            return data, labels

        end_idx = get_masking_index(data)
        max_length = end_idx - self.mask_len_samples
        # TODO: Fix when max_length < 0 (short stages in segment pred)
        start = random.randint(0, max_length)

        t = torch.arange(length, device=data.device).float()
        t = t.repeat(channels, 1)

        s = 1000 / length
        mask = (
            (
                torch.sigmoid(s * -(t - start))
                + torch.sigmoid(s * (t - start - self.mask_len_samples))
            )
            .float()
            .to(data.device)
        )
        data = data * mask.transpose(1, 0)
        labels[start : start + self.mask_len_samples] = 0
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
        end_idx = get_masking_index(data)

        current_label = labels[0]
        sequence_indices = []
        start_idx = 0

        for i in range(1, end_idx):
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

        end_idx = get_masking_index(data, search_value=torch.nan)
        start = random.randint(0, end_idx)
        crop_length = random.randint(5, 15)
        end = start + crop_length
        cropped_data = data[start:end, :]
        cropped_labels = labels[start:end]

        padded_data = torch.full(
            (length, cropped_data.shape[1]), torch.nan, dtype=cropped_data.dtype
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

        end_idx = get_masking_index(data, search_value=torch.nan)
        max_start = max(end_idx - self.CROP_LENGTH, 1)
        beta_sample = torch.distributions.Beta(0.5, 0.5).sample()
        start = int(beta_sample * max_start)
        end = start + self.CROP_LENGTH
        # print(start, end)
        cropped_data = data[start:end, :]
        cropped_labels = labels[start:end, :]

        if cropped_data.shape[0] < self.CROP_LENGTH:
            offset = self.CROP_LENGTH - cropped_data.shape[0]
            cropped_data = torch.nn.functional.pad(
                cropped_data, (0, 0, 0, offset), value=torch.nan
            )
            cropped_labels = torch.nn.functional.pad(
                cropped_labels, (0, 0, 0, offset), value=0
            )
            cropped_labels[-offset:, 0] = 1.0

        return cropped_data, cropped_labels


class ReverseTimeTransform(object):
    def __init__(self, probability=0.5):
        self.probability = probability
        pass

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        # Always use after cropping
        if torch.rand((1,)).item() > self.probability:
            return data, labels

        end_idx = get_masking_index(data, search_value=torch.nan)
        data_flipped = torch.flip(data[:end_idx], dims=[0])
        labels_flipped = torch.flip(labels[:end_idx], dims=[0])
        data[:end_idx] = data_flipped
        labels[:end_idx] = labels_flipped
        # data = torch.flip(data, dims=[0])
        # labels = torch.flip(labels, dims=[0])

        return data, labels


class TimeMaskTransform(object):
    def __init__(self, probability=0.5, mask_length=50):
        self.probability = probability
        self.mask_length = mask_length

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        if torch.rand((1,)).item() > self.probability:
            return data, labels

        end_idx = get_masking_index(data, search_value=torch.nan)
        max_start = max(end_idx - self.mask_length, 1)
        beta_sample = torch.distributions.Beta(0.5, 0.5).sample()
        start = int(beta_sample * max_start)
        end = start + self.mask_length
        # print(start, end)
        data[start:end] = data.mean()
        labels[start:end] = 0
        labels[start:end, 0] = 1.0

        return data, labels
    

class TimeDropoutTransform(object):
    def __init__(self, probability=0.5, mask_length=50):
        self.probability = probability
        self.mask_length = mask_length

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        original_length = data.shape[0]

        if torch.rand((1,)).item() > self.probability:
            return data, labels

        end_idx = get_masking_index(data, search_value=torch.nan)
        max_start = max(end_idx - self.mask_length, 1)

        beta_sample = torch.distributions.Beta(0.5, 0.5).sample()
        start = int(beta_sample * max_start)
        end = start + self.mask_length

        # Cut out the data and stitch the remaining parts together
        data_stitched = torch.cat((data[:start], data[end:]), dim=0)
        labels_stitched = torch.cat((labels[:start], labels[end:]), dim=0)

        padding_size = original_length - data_stitched.shape[0]
        data_padded = torch.nn.functional.pad(data_stitched, (0, 0, 0, padding_size), value=torch.nan)
        labels_padded = torch.nn.functional.pad(labels_stitched, (0, 0, 0, padding_size), value=0.0)
        labels_padded[-padding_size:, 0] = 1.0

        return data_padded, labels_padded


class StartEndMaskTransform(object):
    def __init__(self, probability=0.5, mask_length=50):
        self.probability = probability
        self.mask_length = mask_length

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        if torch.rand((1,)).item() > self.probability:
            return data, labels

        if torch.rand((1,)).item() > 0.5:
            data[-self.mask_length :] = torch.nan
            labels[-self.mask_length :] = 0
            labels[-self.mask_length :, 0] = 1
        else:
            data[: self.mask_length] = torch.nan
            labels[: self.mask_length] = 0
            labels[: self.mask_length, 0] = 1

        return data, labels


class StartJitterTransform(object):
    def __init__(self, offset_before, probability=1.0):
        self.offset_before = offset_before
        self.probability = probability

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        if torch.rand((1,)).item() > self.probability:
            return data, labels

        offset = torch.randint(self.offset_before, (1,))

        cropped_data = data[offset:, :]
        cropped_labels = labels[offset:, :]

        cropped_data = torch.nn.functional.pad(
            cropped_data, (0, 0, 0, offset), value=torch.nan
        )
        cropped_labels = torch.nn.functional.pad(
            cropped_labels, (0, 0, 0, offset), value=0
        )
        cropped_labels[-offset:, 0] = 1.0

        return cropped_data, cropped_labels
    

class EndJitterTransform(object):
    def __init__(self, extra_offset, probability=1.0):
        self.extra_offset = extra_offset
        self.probability = probability

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        if torch.rand((1,)).item() > self.probability:
            return data, labels
        
        offset = torch.randint(self.extra_offset, (1,))

        end_idx = get_masking_index(data, search_value=torch.nan)
        data[end_idx - offset:, :] = torch.nan
        labels[end_idx - offset:, :] = 0
        labels[end_idx - offset:, 0] = 1.0

        return data, labels
    

class StartJitterQuadraticTransform(object):
    def __init__(self, offset_before, probability=1.0):
        self.offset_before = offset_before
        self.probability = probability

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        if torch.rand((1,)).item() > self.probability:
            return data, labels

        offset = int((1 - (torch.rand((1,)) ** 2)) * self.offset_before)

        cropped_data = data[offset:, :]
        cropped_labels = labels[offset:, :]

        cropped_data = torch.nn.functional.pad(
            cropped_data, (0, 0, 0, offset), value=torch.nan
        )
        cropped_labels = torch.nn.functional.pad(
            cropped_labels, (0, 0, 0, offset), value=0
        )
        cropped_labels[-offset:, 0] = 1.0

        return cropped_data, cropped_labels
    

class EndJitterQuadraticTransform(object):
    def __init__(self, extra_offset, probability=1.0):
        self.extra_offset = extra_offset
        self.probability = probability

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]

        if torch.rand((1,)).item() > self.probability:
            return data, labels
        
        offset = int((1 - (torch.rand((1,)) ** 2)) * self.extra_offset)

        end_idx = get_masking_index(data, search_value=torch.nan)
        data[end_idx - offset:, :] = torch.nan
        labels[end_idx - offset:, :] = 0
        labels[end_idx - offset:, 0] = 1.0

        return data, labels

class ConcatenateTransform(object):
    def __init__(self, concat_probability=0.5):
        # Stub class for notifying the generator that concatenation is happening
        self.concat_probability = concat_probability
    
    def __call__(self, data_in):
        return data_in

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
