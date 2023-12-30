from torch import nn
import torch
import torch.nn.functional as F
from hmpai.utilities import MASKING_VALUE
import math
from hmpai.pytorch.utilities import DEVICE


class TransformerModel(nn.Module):
    def __init__(self, n_features, n_heads, ff_dim, n_layers, n_samples, n_classes):
        super().__init__()
        self.pos_encoder = tAPE(n_features, max_len=n_samples)
        self.linear = nn.Linear(n_features, n_features)
        # self.pos_encoder = LearnablePositionalEncoding(n_features, max_len=10)
        # self.pos_encoder = PositionalEncoding(n_features)
        encoder_layers = nn.TransformerEncoderLayer(n_features, n_heads, ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.n_features = n_features
        self.decoder = nn.Linear(n_features, n_classes)

    def forward(self, x):
        # Calculate mask before?
        # values, lengths = torch.max((x == MASKING_VALUE).int(), dim=1)
        # lengths = lengths * values - (1 - values)
        # Why times sqrt(ninp)?
        # x = torch.squeeze(x, dim=1)
        # mask = torch.where(x == MASKING_VALUE, 0.0, 1.0)
        # x = x * math.sqrt(self.n_features)
        mask = (x == MASKING_VALUE).all(dim=2).t()
        x = self.linear(x)
        pos_enc = self.pos_encoder(x)
        # mask = (x == MASKING_VALUE).bool()[:, :, 0].transpose(0, 1)
        x = self.transformer_encoder(pos_enc, src_key_padding_mask=mask)

        inverse_mask = ~mask
        inverse_mask = inverse_mask.float().t().unsqueeze(-1)
        x = x * inverse_mask

        sum_emb = x.sum(dim=1)
        sum_mask = inverse_mask.squeeze(-1).sum(dim=1, keepdim=True)
        mean_pooled = sum_emb / sum_mask.clamp(min=1)

        # x = x * mask
        # x = x.mean(dim=1)
        # mean_pooled = sum_embeddings / sum_mask.clamp(min=1)
        x = self.decoder(mean_pooled)
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

        # distance = torch.matmul(self.pe, self.pe[10])
        # import matplotlib.pyplot as plt

        # plt.plot(distance.detach().numpy())
        # plt.show()

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe
        # x = x + self.pe[: x.size(0), :].to(DEVICE)
        return self.dropout(x)


class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[: x.size(0), :].to(DEVICE)
        # x = x + self.pe
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).to(DEVICE)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :].to(DEVICE)
        return self.dropout(x)


class SAT1Base(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 16 = left over samples after convolutions
        self.linear = nn.LazyLinear(out_features=128)
        self.linear_final = nn.LazyLinear(out_features=n_classes)

        # Kernel order = (samples, channels)
        self.maxpool = nn.MaxPool2d((2, 1))
        self.conv1 = PartialConv2d(in_channels=1, out_channels=64, kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Mask values that are not used from batch
        x = x[:, None, :, :]
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


class SAT1Topological(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 16 = left over samples after convolutions
        self.linear = nn.LazyLinear(out_features=128)
        self.linear_final = nn.LazyLinear(out_features=n_classes)
        # Kernel order = (samples, x, y)
        self.maxpool = nn.MaxPool3d((2, 1, 1))
        self.conv1 = PartialConv3d(
            in_channels=1, out_channels=64, kernel_size=(5, 1, 1)
        )
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 1, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Mask values that are not used from batch
        x = x[:, None, :, :]
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


class SAT1TopologicalConv(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 16, 4, 1 = left over dimensions after convolutions
        self.linear = nn.LazyLinear(out_features=128)
        self.linear_final = nn.LazyLinear(out_features=n_classes)
        # Kernel order = (samples, x, y)
        self.maxpool = nn.MaxPool3d((2, 1, 1))
        self.conv1 = PartialConv3d(
            in_channels=1, out_channels=64, kernel_size=(5, 3, 3)
        )
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3))
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 1, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Mask values that are not used from batch
        # x = torch.squeeze(x, dim=1)
        x = x[:, None, :, :]
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
    def __init__(self, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 19 = left over samples after convolutions
        self.linear = nn.LazyLinear(out_features=512)
        self.linear_final = nn.LazyLinear(out_features=n_classes)
        # Kernel order = (samples, channels)
        self.maxpool = nn.MaxPool2d((2, 1))
        self.conv1 = PartialConv2d(in_channels=1, out_channels=32, kernel_size=(25, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(17, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(11, 1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 1))
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x[:, None, :, :]
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
        # Check shape of x here to determine # samples
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
        # self.gru = nn.GRU(input_size=n_channels, hidden_size=16, batch_first=True, dropout=0.25)
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=256,
            batch_first=True,
        )
        self.linear = nn.LazyLinear(out_features=128)
        self.linear_final = nn.LazyLinear(out_features=n_classes)

    def forward(self, x):
        # Shape = [batch_size, 1, samples, channels]
        x = torch.squeeze(x, dim=1)
        samples = x.shape[1]
        # Find lengths of sequences
        values, lengths = torch.max((x == MASKING_VALUE).int(), dim=1)
        lengths = lengths * values - (1 - values)
        lengths = lengths.masked_fill_(values == 0, samples)
        # Goes wrong when dims is 1? When does this happen
        lengths = lengths[:, 0]
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), enforce_sorted=False, batch_first=True
        )
        x, _ = self.gru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, padding_value=MASKING_VALUE, total_length=samples
        )
        x = self.relu(x)
        x = self.linear(x)
        x = self.linear_final(x)

        # Select indices of last sample
        lengths = torch.as_tensor(lengths) - 1
        batch_size = x.shape[0]
        batch_indices = torch.arange(batch_size)
        x = x.transpose(0, 1)[lengths, batch_indices, :]

        return x


class SAT1LSTM(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.gru = nn.LSTM(input_size=n_channels, hidden_size=256, batch_first=True)
        self.linear = nn.LazyLinear(out_features=128)
        self.linear_final = nn.LazyLinear(out_features=n_classes)

    def forward(self, x):
        x = torch.squeeze(x)
        samples = x.shape[1]
        # Find lengths of sequences
        values, lengths = torch.max((x == MASKING_VALUE).int(), dim=1)
        lengths = lengths * values - (1 - values)
        lengths = lengths.masked_fill_(values == 0, samples)
        lengths = lengths[:, 0]
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), enforce_sorted=False, batch_first=True
        )
        x, _ = self.gru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, padding_value=MASKING_VALUE, total_length=samples
        )
        x = self.relu(x)
        x = self.linear(x)
        x = self.linear_final(x)

        # Select indices of last sample
        lengths = torch.as_tensor(lengths) - 1
        batch_size = x.shape[0]
        batch_indices = torch.arange(batch_size)
        x = x.transpose(0, 1)[lengths, batch_indices, :]

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


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
class PartialConv3d(nn.Conv3d):
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

        super(PartialConv3d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
            * self.weight_maskUpdater.shape[4]
        )

        self.last_size = (None, None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 5
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
                            input.data.shape[4],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1,
                            1,
                            input.data.shape[2],
                            input.data.shape[3],
                            input.data.shape[4],
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv3d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv3d, self).forward(
            torch.mul(input, mask_in) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output
