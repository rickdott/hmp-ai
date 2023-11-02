from torch import nn
import torch
import torch.nn.functional as F
from hmpai.utilities import MASKING_VALUE


class SAT1Mlp(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=n_channels * n_samples, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=32)
        self.linear_final = nn.Linear(in_features=32, out_features=n_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear_final(x)

        return x


class SAT1Base(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 16 = left over samples after convolutions
        self.linear = nn.Linear(in_features=256 * 16 * n_channels, out_features=128)
        self.linear_final = nn.Linear(in_features=128, out_features=n_classes)
        # Kernel order = (samples, channels)
        self.maxpool = nn.MaxPool2d((2, 1))
        self.conv1 = PartialConv2d(in_channels=1, out_channels=64, kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Mask values that are not used from batch
        mask_in = torch.where(x == MASKING_VALUE, 0.0, 1.0)
        x = self.conv1(x, mask_in=mask_in)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_final(x)

        return x


class SAT1Deep(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 17 = left over samples after convolutions
        self.linear = nn.Linear(in_features=1024 * 17 * n_channels, out_features=512)
        self.linear_final = nn.Linear(in_features=512, out_features=n_classes)
        # Kernel order = (samples, channels)
        self.maxpool = nn.MaxPool2d((2, 1))
        self.conv1 = PartialConv2d(in_channels=1, out_channels=64, kernel_size=(25, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(17, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(11, 1))
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 1))
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        mask_in = torch.where(x == MASKING_VALUE, 0.0, 1.0)
        x = self.conv1(x, mask_in=mask_in)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_final(x)

        return x


class SAT1GRU(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=30, hidden_size=256, batch_first=True)
        self.linear = nn.Linear(in_features=256, out_features=128)
        self.linear_final = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x):
        x = torch.squeeze(x)
        mask_in = torch.where(x == MASKING_VALUE, 0.0, 1.0)
        x, _ = self.gru(x)
        x = self.relu(x)
        x = x * mask_in[:, :, 0].unsqueeze(-1)
        x = self.linear(x)
        x = self.linear_final(x)

        return x


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
        )

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(input, mask) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output
