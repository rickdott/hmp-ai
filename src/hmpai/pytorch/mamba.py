import torch
from torch import nn
from mamba_ssm import Mamba2, Mamba
from hmpai.utilities import MASKING_VALUE, get_masking_indices
import numpy as np


def build_mamba(config):
    class MambaModel(nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.config = config

            if "n_channels" not in config:
                raise ValueError("Config must contain 'n_channels' key")
            self.n_channels = config.get("n_channels")

            if "n_mamba_layers" not in config:
                raise ValueError("Config must contain 'n_mamba_layers' key")
            n_mamba_layers = config.get("n_mamba_layers")

            if "n_classes" not in config:
                raise ValueError("Config must contain 'n_classes' key")
            n_classes = config.get("n_classes")

            self.use_pos_enc = config.get("use_pos_enc", False)

            # Spatial feature extraction
            if "spatial_feature_dim" not in config and (
                "use_linear_fe" in config or "use_pointconv_fe" in config
            ):
                raise ValueError(
                    "'spatial_feature_dim' must be provided if any feature extraction is done"
                )
            if config.get("use_linear_fe", False):
                self.spatial_feature_dim = config.get("spatial_feature_dim")
                self.spatial_layer = nn.Linear(
                    in_features=self.n_channels, out_features=self.spatial_feature_dim
                )
            elif config.get("use_pointconv_fe", False):
                self.spatial_feature_dim = config.get("spatial_feature_dim")
                self.spatial_layer = nn.Conv1d(
                    in_channels=self.n_channels,
                    out_channels=self.spatial_feature_dim,
                    kernel_size=1,
                )
            else:
                # Dummy layer that does nothing
                self.spatial_feature_dim = self.n_channels
                self.spatial_layer = nn.Identity()

            self.temporal_dropout = nn.Dropout1d(p=0.2)
            self.activation = nn.SiLU()

            # Convolutional layers
            if config.get("use_conv", False):
                if "conv_kernel_sizes" not in config:
                    raise ValueError(
                        "If 'use_conv' is True, 'conv_kernel_sizes' must be provided"
                    )
                self.conv_kernel_sizes = config.get("conv_kernel_sizes")
                if "conv_in_channels" not in config:
                    raise ValueError(
                        "If 'use_conv' is True, 'conv_in_channels' must be provided"
                    )
                self.conv_in_channels = config.get("conv_in_channels")
                if "conv_out_channels" not in config:
                    raise ValueError(
                        "If 'use_conv' is True, 'conv_out_channels' must be provided"
                    )
                self.conv_out_channels = config.get("conv_out_channels")

                if config.get("conv_stack", False):
                    conv_modules = []
                    for kernel_size, in_channels, out_channels in zip(
                        self.conv_kernel_sizes,
                        self.conv_in_channels,
                        self.conv_out_channels,
                    ):
                        conv_modules.append(
                            nn.Conv1d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding="same",
                            )
                        )
                    self.temporal_module = nn.Sequential(*conv_modules)
                elif config.get("conv_concat", False):

                    class ConcatConv(nn.Module):
                        def __init__(
                            self, conv_kernel_sizes, conv_in_channels, conv_out_channels
                        ):
                            super().__init__()
                            self.conv_layers = nn.ModuleList()
                            for kernel_size, in_channels, out_channels in zip(
                                conv_kernel_sizes, conv_in_channels, conv_out_channels
                            ):
                                self.conv_layers.append(
                                    nn.Conv1d(
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        padding="same",
                                    )
                                )

                        def forward(self, x):
                            tmp_results = []
                            for conv_layer in self.conv_layers:
                                tmp_results.append(conv_layer(x))
                            return torch.cat(tmp_results, dim=1)

                    self.temporal_module = ConcatConv(
                        self.conv_kernel_sizes,
                        self.conv_in_channels,
                        self.conv_out_channels,
                    )
                else:
                    raise ValueError(
                        "If 'use_conv' is True, either 'conv_stack' or 'conv_concat' must be True"
                    )
            else:
                self.temporal_module = nn.Identity()

            mamba_dim = self.__calculate_mamba_dim__()
            if config.get("use_lstm", False):
                self.seq_model = nn.Sequential(
                    *[LSTMBlock(mamba_dim) for _ in range(n_mamba_layers)]
                )
            else:
                self.seq_model = nn.Sequential(
                    *[MambaBlock(mamba_dim) for _ in range(n_mamba_layers)]
                )
            self.normalization = nn.LayerNorm(mamba_dim)
            self.linear_out = nn.Linear(mamba_dim, n_classes)

        def __calculate_mamba_dim__(self):
            mamba_dim = self.spatial_feature_dim
            if config.get("use_conv", False):
                if config.get("conv_stack", False):
                    # Feature dim will equal out_channels of last conv layer
                    mamba_dim = self.conv_out_channels[-1]
                elif config.get("conv_concat", False):
                    # Feature dim will equal sum of all out_channels
                    mamba_dim = np.sum(self.conv_out_channels).item()
            if self.use_pos_enc:
                mamba_dim += 1

            return mamba_dim

        def forward(self, x):
            if self.use_pos_enc:
                if x.shape[-1] != self.n_channels + 1:
                    raise ValueError(
                        "Positional encoding was likely not supplied as an extra feature, check input"
                    )
                pe = x[..., -1].unsqueeze(-1)
                x = x[..., :-1]

            max_indices = get_masking_indices(x)
            max_seq_len = max_indices.max()

            if self.use_pos_enc:
                pe = pe[:, :max_seq_len]

            x = x[:, :max_seq_len, :]

            # Spatial
            if self.config.get("use_linear_fe", False):
                x = self.spatial_layer(x)
                x = x.permute(0, 2, 1)
                x = self.temporal_dropout(x)
                x = self.activation(x)
            elif self.config.get("use_pointconv_fe", False):
                x = x.permute(0, 2, 1)
                x = self.spatial_layer(x)
                x = self.temporal_dropout(x)
                x = self.activation(x)
            else:
                # Permutation is still necessary for next step, if use_conv is false it will just be reversed by the next permute call
                x = x.permute(0, 2, 1)

            # Temporal
            if self.config.get("use_conv", False):
                x = self.temporal_module(x)
                x = self.activation(x)

            x = x.permute(0, 2, 1)

            if self.use_pos_enc:
                # Append positional encoding feature
                x = torch.cat([x, pe], dim=-1)

            x = self.seq_model(x)
            x = self.normalization(x)
            x = self.linear_out(x)

            return x

    return MambaModel(config)


class MambaBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mamba = Mamba(d_model=embed_dim, d_state=64, d_conv=4, expand=2)
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        x = self.mamba(self.norm(x)) + x
        return x


class LSTMBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        x = self.lstm(self.norm(x))[0] + x
        return x
