import torch
from hmpai.utilities import get_masking_index


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
        data[end_idx - offset :, :] = torch.nan
        labels[end_idx - offset :, :] = 0
        labels[end_idx - offset :, 0] = 1.0

        return data, labels
