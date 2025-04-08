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


class AblationMamba(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        global_pool: bool = False,
        config: dict = {},  # Contains flags/values for ablation, if key is not present, dont add ablation
    ):
        super().__init__()
        self.n_channels = n_channels
        n_mamba_layers = config.get("mamba_layers")

        # Linear embedding from feature space (n_channels), to embed_dim
        # self.linear_in = nn.Linear(n_channels, embed_dim)

        self.pointconv = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        # self.conv_in = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(n_channels, 1), padding=0)

        self.pos_enc = ReactionTimeRelativeEncoding()
        # self.pos_enc = NormalizedRelativePositionalEncoding(mamba_dim)
        self.conv1 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=9, padding="same"
        )
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=27, padding="same"
        )

        self.norm = nn.LayerNorm(mamba_dim)

        # Define sequence of Mamba blocks
        # mamba_dim should be the feature length of output, so it depends on which ablations are used
        self.blocks = nn.Sequential(
            *[MambaBlock(mamba_dim) for _ in range(n_mamba_layers)]
        )
        self.global_pool = global_pool
        self.linear_translation = nn.Linear(mamba_dim, n_channels)
        self.linear_out = nn.Linear(mamba_dim, n_classes)
        self.pretraining = False

    def forward(self, x):
        pe = None
        if x.shape[-1] == self.n_channels + 1:
            pe = x[:, :, -1].unsqueeze(-1)
            x = x[:, :, :-1]

        max_indices = get_masking_indices(x)

        max_seq_len = max_indices.max()

        if pe is not None:
            pe = pe[:, :max_seq_len]

        # [64, 634, 19]
        # [B, T, C]
        x = x[:, :max_seq_len, :]
        # [64, 634-max_index, 19]
        # [B, T-max_index, 19]
        # Linear with same size to learn relationships between electrodes
        # x = self.linear_in(x)

        # Linear (64 > 64), Conv (64 > 128)
        # 1D point-wise convolution to convert 64 channels into 64 features, including correlation across electrodes
        x = x.permute(0, 2, 1)
        x = self.pointconv(x)
        # Dropout on time dimension to decrease dependence on temporal information
        x = nn.Dropout1d(p=0.2)(x)
        x = nn.functional.silu(x)

        # Three conv layers with gradually increasing kernel sizes to capture temporal relationships at different time scales
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x1
        x = torch.cat([x1, x2, x3], dim=1)
        x = nn.functional.silu(x)

        x = x.permute(0, 2, 1)

        x = torch.cat([x, pe], dim=-1)

        out_forward = (
            self.blocks(x)
            if not self.global_pool
            else torch.mean(self.blocks(x), dim=1)
        )

        out = out_forward
        out = self.norm(out)
        out = (
            self.linear_out(out)
            if not self.pretraining
            else self.linear_translation(out)
        )
        return out


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


class ReactionTimeRelativeEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, max_indices):
        max_seq_len = max_indices.max()
        batch_size = len(max_indices)

        positional_encodings = (
            torch.arange(max_seq_len, device=max_indices.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .float()
        )
        normalized_positions = (
            positional_encodings / (max_indices.unsqueeze(1)).clamp(min=1)
        ).clamp(
            max=1
        )  # Normalize positions over range [0, 1] up to max_idx

        return normalized_positions.unsqueeze(-1)


class NormalizedRelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Learnable embedding for relative distances in normalized space
        self.relative_embedding = nn.Linear(1, embed_dim)

    def forward(self, seq_len):
        max_seq_len = seq_len.max()
        # Create normalized relative positions (normalized to [0, 1])
        normalized_positions = torch.arange(
            max_seq_len, device=seq_len.device
        ).unsqueeze(0).float() / (max_seq_len - 1)
        relative_positions = normalized_positions.unsqueeze(
            -1
        ) - normalized_positions.unsqueeze(-2)
        relative_positions = (
            relative_positions.abs()
        )  # Absolute difference for relationships

        # Apply embedding to the relative positions
        embedded_positions = self.relative_embedding(
            relative_positions.unsqueeze(-1)
        )  # Shape: (seq_len, seq_len, embed_dim)
        return embedded_positions
