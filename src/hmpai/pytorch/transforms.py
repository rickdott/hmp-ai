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
    def __init__(self, offset_before=0, probability=1.0):
        self.offset_before = offset_before
        self.probability = probability

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]
        context = data_in[2] if len(data_in) > 2 else None

        if torch.rand((1,)).item() > self.probability:
            return data, labels, context

        offset_before = context['start_jitter'] if context and 'start_jitter' in context else self.offset_before
        offset = torch.randint(offset_before, (1,))

        cropped_data = data[offset:, :]
        cropped_labels = labels[offset:, :]

        cropped_data = torch.nn.functional.pad(
            cropped_data, (0, 0, 0, offset), value=torch.nan
        )
        cropped_labels = torch.nn.functional.pad(
            cropped_labels, (0, 0, 0, offset), value=0
        )
        cropped_labels[-offset:, 0] = 1.0

        return cropped_data, cropped_labels, context


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
    def __init__(self, extra_offset=0, probability=1.0):
        self.extra_offset = extra_offset
        self.probability = probability

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]
        context = data_in[2] if len(data_in) > 2 else None

        if torch.rand((1,)).item() > self.probability:
            return data, labels, context

        extra_offset = context['end_jitter'] if context and 'end_jitter' in context else self.extra_offset
        offset = torch.randint(extra_offset, (1,))

        end_idx = get_masking_index(data, search_value=torch.nan)
        data[end_idx - offset :, :] = torch.nan
        labels[end_idx - offset :, :] = 0
        labels[end_idx - offset :, 0] = 1.0

        return data, labels, context


class ChannelShuffleTransform(object):
    """
    A PyTorch transform that randomly shuffles the channels of the input data 
    with a specified probability. This is useful for data augmentation in scenarios 
    where channel order should not affect the model's performance.

    Attributes:
        probability (float): The probability of applying the channel shuffle. 
                             Defaults to 1.0.

    Methods:
        __call__(data):
            Applies the channel shuffle to the input data.

    Args:
        data (torch.Tensor): The input data tensor of shape (T, C), where T is 
                             the number of time steps and C is the number of channels.

    Returns:
        torch.Tensor: The transformed data tensor with channels shuffled, or 
                      the original data if the transform is not applied.
    """
    def __init__(self, probability=1.0):
        self.probability = probability

    def __call__(self, data_in):
        data = data_in[0]
        labels = data_in[1]
        context = data_in[2] if len(data_in) > 2 else None
        
        if torch.rand((1,)).item() > self.probability:
            return data, labels, context

        n_channels = data.shape[1]
        perm = torch.randperm(n_channels)
        if context is not None:
            context['perm'] = perm
        shuffled_data = data[:, perm]

        return shuffled_data, labels, context