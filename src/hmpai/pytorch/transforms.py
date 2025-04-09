import torch
from hmpai.utilities import get_masking_index


class StartJitterTransform(object):
    """
    Applies a random temporal jitter to the start of input data with a given probability.

    Attributes:
        offset_before (int): Maximum offset for jitter.
        probability (float): Probability of applying jitter (default: 1.0).

    Args:
        data_in (tuple): A tuple of:
            - data (torch.Tensor): Input data of shape (T, F).
            - labels (torch.Tensor): Corresponding labels of shape (T, L).

    Returns:
        tuple: Transformed (data, labels) with jitter applied or unchanged if skipped.

    Notes:
        - Pads data with NaN and labels with 0 to maintain size.
        - Sets the last `offset` elements of the first label column to 1.0.
    """
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
    """
    A PyTorch transform that modifies the input data and labels by introducing 
    a random offset near the end of the sequence. This is useful for augmenting 
    data by simulating variations in sequence endings.

    Attributes:
        extra_offset (int): The maximum offset to apply near the end of the sequence.
        probability (float): The probability of applying the transform. Defaults to 1.0.

    Methods:
        __call__(data_in):
            Applies the transform to the input data and labels.

    Args:
        data_in (tuple): A tuple containing:
            - data (torch.Tensor): The input data tensor.
            - labels (torch.Tensor): The corresponding labels tensor.

    Returns:
        tuple: A tuple containing the modified data and labels tensors. If the 
        transform is not applied (based on the probability), the input data and 
        labels are returned unchanged.

    Notes:
        - The transform identifies the end index of the sequence using a masking 
          value (e.g., `torch.nan`).
        - The data and labels are modified starting from the calculated end index 
          minus a random offset.
        - The labels are updated such that the last modified position is set to 1.0 
          in the first dimension, and the rest are set to 0.
    """
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
