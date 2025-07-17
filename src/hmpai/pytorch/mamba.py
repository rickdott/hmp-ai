from hmpai.pytorch.utilities import TASKS
import torch
from torch import nn
from mamba_ssm import Mamba
from hmpai.utilities import get_masking_indices
import numpy as np


def build_mamba(config):
    """
    build_mamba(config)

    Builds a configurable PyTorch model called `MambaModel` based on the provided configuration dictionary.

    Parameters:
        config (dict): A dictionary containing the configuration for the model. The following keys are expected:
            - n_channels (int): Number of input channels. (Required)
            - n_mamba_layers (int): Number of Mamba or LSTM layers. (Required)
            - n_classes (int): Number of output classes. (Required)
            - use_pos_enc (bool, optional): Whether to use positional encoding. Defaults to False.
            - spatial_feature_dim (int, optional): Dimension of spatial features. Required if `use_linear_fe` or `use_pointconv_fe` is True.
            - use_linear_fe (bool, optional): Whether to use a linear layer for spatial feature extraction. Defaults to False.
            - use_pointconv_fe (bool, optional): Whether to use a 1D convolutional layer for spatial feature extraction. Defaults to False.
            - use_conv (bool, optional): Whether to use convolutional layers for temporal feature extraction. Defaults to False.
            - conv_kernel_sizes (list of int, optional): Kernel sizes for convolutional layers. Required if `use_conv` is True.
            - conv_in_channels (list of int, optional): Input channels for convolutional layers. Required if `use_conv` is True.
            - conv_out_channels (list of int, optional): Output channels for convolutional layers. Required if `use_conv` is True.
            - conv_stack (bool, optional): Whether to stack convolutional layers sequentially. Defaults to False.
            - conv_concat (bool, optional): Whether to concatenate outputs of convolutional layers. Defaults to False.
            - use_lstm (bool, optional): Whether to use LSTM layers instead of Mamba layers. Defaults to False.

    Returns:
        nn.Module: A PyTorch model instance configured based on the provided `config`.

    Raises:
        ValueError: If required keys are missing from the `config` dictionary or if incompatible configurations are provided.

    Notes:
        - The model supports optional spatial feature extraction using either a linear layer or a 1D convolutional layer.
        - Temporal feature extraction can be performed using stacked or concatenated convolutional layers.
        - The model can use either Mamba layers or LSTM layers for sequence modeling.
        - Positional encoding can be optionally included as an additional feature.
    """
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

            self.mamba_dim = self.__calculate_mamba_dim__()
            if config.get("use_lstm", False):
                self.seq_model = nn.Sequential(
                    *[LSTMBlock(self.mamba_dim) for _ in range(n_mamba_layers)]
                )
            else:
                self.seq_model = nn.Sequential(
                    *[MambaBlock(self.mamba_dim) for _ in range(n_mamba_layers)]
                )
            self.normalization = nn.LayerNorm(self.mamba_dim)
            self.linear_out = nn.Linear(self.mamba_dim, n_classes)
            self.classification_head = nn.ModuleDict({
                task_name: nn.Linear(self.mamba_dim, n_classes)
                for task_name in TASKS.keys()
            })

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

        def forward(self, x, return_embeddings=False, task=None):
            if self.use_pos_enc:
                if x.shape[-1] != self.n_channels + 1 and x.shape[-1] != self.n_channels + 2:
                    raise ValueError(
                        "Positional encoding was likely not supplied as an extra feature, check input"
                    )
                if x.shape[-1] == self.n_channels + 2:
                    # Positional encoding is the last two features
                    pe = x[..., -2:]
                    x = x[..., :-2]
                elif x.shape[-1] == self.n_channels + 1:
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

            if return_embeddings:
                emb = x.clone()
            if task is None:
                x = self.linear_out(x)
            else:
                x = torch.stack([
                    self.classification_head[t](x[i, :, :]) for i, t in enumerate(task)
                ])

            if return_embeddings:
                return x, emb
            return x

    return MambaModel(config)


class MambaBlock(nn.Module):
    """
    A PyTorch module that applies a Mamba block followed by RMS normalization.

    The MambaBlock is designed to process input tensors using a combination of
    a custom Mamba layer and RMS normalization. The output of the Mamba layer
    is added to the input tensor (residual connection) to enhance gradient flow
    and improve training stability.

    Attributes:
        mamba (Mamba): A custom Mamba layer initialized with specific parameters
            such as embedding dimension, state size, convolution size, and expansion factor.
        norm (nn.RMSNorm): An RMS normalization layer applied to the input tensor.

    Args:
        embed_dim (int): The embedding dimension of the input tensor.

    Methods:
        forward(x):
            Applies the Mamba layer and RMS normalization to the input tensor,
            followed by a residual connection.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, seq_len, embed_dim).

            Returns:
                torch.Tensor: The output tensor of the same shape as the input.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.mamba = Mamba(d_model=embed_dim, d_state=64, d_conv=4, expand=2)
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        x = self.mamba(self.norm(x)) + x
        return x


class LSTMBlock(nn.Module):
    """
    LSTMBlock is a neural network module that combines an LSTM layer with RMS normalization.

    This block applies RMS normalization to the input, processes it through an LSTM layer,
    and adds the original input to the output of the LSTM layer (residual connection).

    Attributes:
        lstm (nn.LSTM): The LSTM layer with input and hidden dimensions equal to `embed_dim`.
        norm (nn.RMSNorm): The RMS normalization layer applied to the input.

    Args:
        embed_dim (int): The dimensionality of the input and hidden states of the LSTM layer.

    Forward Pass:
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim) after
            applying RMS normalization, LSTM, and residual connection.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        x = self.lstm(self.norm(x))[0] + x
        return x
