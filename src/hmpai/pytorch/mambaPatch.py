from hmpai.pytorch.utilities import TASKS
import torch
from torch import nn
from mamba_ssm import Mamba
from hmpai.utilities import get_masking_indices
import numpy as np


def build_mamba_patch(config):
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
    class MambaModelPatch(nn.Module):
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
            if "spatial_feature_dim" not in config:
                raise ValueError(
                    "'spatial_feature_dim' must be provided"
                )
            self.spatial_feature_dim = config.get("spatial_feature_dim")
            self.patch_size = config.get("spatial_patch_size", 50)
            self.spatial_layer = SpatialFeatureExtractor(embed_dim=self.spatial_feature_dim, patch_size=self.patch_size)

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

            self.mamba_dim = self.__calculate_mamba_dim__()
            if config.get("use_lstm", False):
                self.seq_model = nn.Sequential(
                    *[LSTMBlock(self.mamba_dim) for _ in range(n_mamba_layers)]
                )
            else:
                # Alternating forward and backward Mamba layers
                layers = []
                for i in range(n_mamba_layers):
                    # Alternate between forward (even indices) and backward (odd indices)
                    is_backward = (i % 2 == 1)
                    layers.append(MambaBlock(self.mamba_dim, backward=is_backward))
                self.seq_model = nn.Sequential(*layers)
            self.normalization = nn.LayerNorm(self.mamba_dim)
            self.linear_out = nn.Linear(self.mamba_dim, n_classes)
            
            self.classification_prep = ClassificationPrep(
                self.mamba_dim, 
                self.patch_size, 
                n_classes
            )
            
            self.classification_head = nn.ModuleDict({
                task_name: ClassificationHead(
                    emb_dim=self.mamba_dim,
                    n_classes=n_classes,
                    n_channels=self.n_channels,
                    patch_size=self.patch_size
                ) for task_name in TASKS.keys()
            })

        def __calculate_mamba_dim__(self):
            mamba_dim = self.spatial_feature_dim
            if self.use_pos_enc:
                mamba_dim += 2
            return mamba_dim

        def forward(self, x, return_embeddings=False, task=None):
            max_indices = get_masking_indices(x)
            max_seq_len = max_indices.max()

            x = x[:, :max_seq_len, :]

            x = x.permute(0, 2, 1)
            x = self.spatial_layer(x)

            x = self.seq_model(x)
            x = self.normalization(x)
            x = self.classification_prep(x, max_seq_len)

            emb = x.clone() if return_embeddings else None
            
            if task is None:
                x = self.linear_out(x)
            else:
                x = torch.stack([
                    self.classification_head[t](x[i, :, :]) for i, t in enumerate(task)
                ])

            if return_embeddings:
                return x, emb
            return x

    return MambaModelPatch(config)


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
        backward (bool): Whether to process the sequence in reverse order.

    Args:
        embed_dim (int): The embedding dimension of the input tensor.
        backward (bool, optional): Whether to reverse the sequence before processing. Defaults to False.

    Methods:
        forward(x):
            Applies the Mamba layer and RMS normalization to the input tensor,
            followed by a residual connection. If backward=True, reverses the sequence
            before and after processing.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, seq_len, embed_dim).

            Returns:
                torch.Tensor: The output tensor of the same shape as the input.
    """
    def __init__(self, embed_dim, backward=False):
        super().__init__()
        self.mamba = Mamba(d_model=embed_dim, d_state=64, d_conv=4, expand=2)
        self.norm = nn.RMSNorm(embed_dim)
        self.backward = backward

    def forward(self, x):
        if self.backward:
            # Process sequence in reverse order
            x = torch.flip(x, [1])
            x = self.mamba(self.norm(x)) + x
            x = torch.flip(x, [1])
        else:
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

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, patch_size=13):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(embed_dim)

        self.time_module = TimeModule(embed_dim, groups=8, patch_size=patch_size)
        self.spectral_module = SpectralModule(embed_dim, patch_size=patch_size)
        self.positional_module = PositionalModule(embed_dim, patch_size=patch_size)

    def forward(self, x):
        # Split into patches
        # Input (B, C, T)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size) # (B, C, T//patch_size, patch_size)
        # Remove PE (last channel)
        pe = x[:, -1:, :, :]
        x = x[:, :-1, :, :]

        x_time = self.time_module(x)
        # x_spectral = self.spectral_module(x)
        x_total = x_time #+ x_spectral

        # Conditional Positional Encoding
        x_pos = self.positional_module(x_total)

        x_total = x_total + x_pos
        pe = torch.cat([pe.min(dim=-1, keepdims=True).values, pe.max(dim=-1, keepdims=True).values], dim=-1)
        pe = pe.expand(-1, x_total.shape[1], -1, -1)

        x_total = torch.cat([x_total, pe], dim=-1)

        B, C, n, D = x_total.shape
        x_total = x_total.reshape(B, C*n, D)
        return x_total


class TimeModule(nn.Module):
    def __init__(self, embed_dim, groups=8, patch_size=13):
        super().__init__()
        self.embed_dim = embed_dim
        self.groups = groups
        self.patch_size = patch_size

        self.conv = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=1, padding=0)
        self.norm = nn.GroupNorm(num_groups=embed_dim // groups, num_channels=embed_dim)
        self.act = nn.GELU()
        # TODO: Figure out how to add residual connection, this doesnt work
        self.shortcut = nn.Conv1d(1, embed_dim, kernel_size=1)


    def forward(self, x):
        # (B, C, T//patch_size, patch_size)
        B, C, n, t = x.shape
        x = x.reshape(B*C*n, 1, t)
        # res = self.shortcut(x)

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        # x += res

        x = x.reshape(B, C, n, -1)
        return x
    
class SpectralModule(nn.Module):
    def __init__(self, embed_dim, patch_size=13):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(nn.Linear((patch_size // 2) + 1, self.embed_dim), nn.Dropout(0.1))
    
    def forward(self, x):
        B, C, n, t = x.shape
        x = x.reshape(B*C*n, t).contiguous()
        spectral = torch.fft.rfft(x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(B, C, n, -1)
        spectral_emb = self.proj(spectral)
        return spectral_emb


class PositionalModule(nn.Module):
    def __init__(self, embed_dim, patch_size=13):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # Conditional Positional Encoding, should encode spatial dimension
        # (3, 1) = (space, n_patches)
        # self.proj = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=(19, 7), stride=1, padding=(9, 3), bias=True, groups=self.embed_dim)
        self.proj = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True, groups=self.embed_dim)

    def forward(self, x):
        # x: (B, C, n, D)
        x = x.permute(0, 3, 1, 2)
        # x: B, D, C, n
        # Dont add residual here since this is added to orig vector anyway
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        # x: B, C, n, D
        return x


class ClassificationPrep(nn.Module):
    def __init__(self, emb_dim, patch_size, n_classes):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.n_classes = n_classes
        # Original single-step approach
        self.head = nn.ConvTranspose1d(
            in_channels=self.emb_dim, 
            out_channels=self.emb_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size, 
            padding=0
        )
        self.refine = nn.Sequential(
            nn.Conv1d(self.emb_dim, self.emb_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.emb_dim, self.emb_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, max_seq_len):
        B, L, D = x.shape
        # L = C * n_patches
        # C = n_channels
        n_patches = max_seq_len // self.patch_size

        x = x.view(B, -1, n_patches, D)
        # Mean over channels
        x = x.mean(dim=1)

        x = x.transpose(1, -1)  # (B, D, C*n_patches)
        y = self.head(x)  # (B, D, C*n_patches * patch_size)
        y = self.refine(y)

        return y
    
    def _get_num_groups(self, num_channels):
        """Find a valid number of groups that divides num_channels"""
        # Try preferred group sizes in descending order
        for groups in [32, 16, 8, 4, 2, 1]:
            if num_channels % groups == 0:
                return groups
        return 1  # Fallback to 1 group (equivalent to LayerNorm)

class ClassificationHead(nn.Module):
    def __init__(self, emb_dim, n_classes, n_channels=63, patch_size=50):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        
        self.classifier = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(emb_dim // 2, n_classes, 1)
        )
    
    def forward(self, x):
        # x: (D, C*T) where C=n_channels, T=time_steps
        x = x.unsqueeze(0)  # (1, D, C*T)
        
        # Classify each time step
        y = self.classifier(x)  # (1, n_classes, T)
        y = y.squeeze(0).transpose(1, 0)  # (T, n_classes)

        return y