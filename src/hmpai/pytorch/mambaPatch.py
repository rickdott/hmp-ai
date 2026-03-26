import torch
from torch import nn
from mamba_ssm import Mamba, Mamba2
from hmpai.utilities import get_masking_indices, MASKING_VALUE
import numpy as np


def build_mamba_patch(config):
    """
    build_mamba(config)

    Builds a configurable PyTorch model called `MambaModel` based on the provided configuration dictionary.

    Parameters:
        config (dict): A dictionary containing the configuration for the model. The following keys are expected:
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

            if "task_class_counts" not in config:
                raise ValueError("Config must contain 'task_class_counts' key")
            self.task_class_counts = config.get("task_class_counts")
            if "n_mamba_layers" not in config:
                raise ValueError("Config must contain 'n_mamba_layers' key")
            n_mamba_layers = config.get("n_mamba_layers")

            self.use_pos_enc = config.get("use_pos_enc", False)

            # Spatial feature extraction
            if "spatial_feature_dim" not in config:
                raise ValueError(
                    "'spatial_feature_dim' must be provided"
                )
            self.spatial_feature_dim = config.get("spatial_feature_dim")
            self.patch_size = config.get("spatial_patch_size", 50)
            self.feature_extractor = FeatureExtractor(embed_dim=self.spatial_feature_dim, patch_size=self.patch_size, use_pos_enc=self.use_pos_enc)

            self.mamba_dim = self.__calculate_mamba_dim__()
            if config.get("use_lstm", False):
                self.seq_model = nn.Sequential(
                    *[LSTMBlock(self.mamba_dim) for _ in range(n_mamba_layers)]
                )
            else:
                layers = []
                for _ in range(n_mamba_layers):
                    layers.append(MambaBlock(self.mamba_dim))
                self.seq_model = nn.Sequential(*layers)
            self.normalization = nn.LayerNorm(self.mamba_dim)
            self.linear_out = nn.Linear(self.mamba_dim, sum(self.task_class_counts.values()))

            self.classification_prep = ClassificationPrep(
                self.mamba_dim, 
                self.patch_size
            )

            self.spatial_attention = nn.Sequential(
                nn.Conv2d(self.mamba_dim, self.mamba_dim // 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.Conv2d(self.mamba_dim // 4, 1, kernel_size=(1, 1)),
            )

            self.classification_head = nn.ModuleDict({
                t: ClassificationHead(emb_dim=self.mamba_dim, n_classes=n)
                for t, n in self.task_class_counts.items()
            })

        def __calculate_mamba_dim__(self):
            mamba_dim = self.spatial_feature_dim
            # # TODO: Change if pos_enc is embedded into embedding
            # if self.use_pos_enc:
            #     mamba_dim += 2
            return mamba_dim

        def forward(self, x, return_embeddings=False, task=None, coords=None):
            max_indices = get_masking_indices(x)
            max_seq_len = max_indices.max()

            x = x[:, :max_seq_len, :]

            x = x.permute(0, 2, 1)
            x, ch_mask = self.feature_extractor(x, coords)

            x = self.seq_model(x)
            x = self.normalization(x)
            x = self.classification_prep(x, max_seq_len)
            x_weights = x.permute(0, 2, 1, 3)
            weights = self.spatial_attention(x_weights)
            weights = weights.masked_fill(~ch_mask.unsqueeze(1), float('-inf'))
            weights = torch.softmax(weights, dim=2)
            x = (x_weights*weights).sum(dim=2)

            emb = x.clone() if return_embeddings else None
            
            if task is None:
                x = self.linear_out(x)
            else:
                # Compute outputs for each sample
                outputs = []
                for i, t in enumerate(task):
                    out = self.classification_head[t](x[i, :, :])  # (T, n_classes_for_task_t)
                    outputs.append(out)
                
                # Find max number of classes in this batch
                max_classes = max(out.shape[-1] for out in outputs)
                
                # Pad outputs to max_classes
                padded_outputs = []
                for out in outputs:
                    if out.shape[-1] < max_classes:
                        # Pad with MASKING_VALUE
                        pad_size = max_classes - out.shape[-1]
                        out_padded = torch.nn.functional.pad(out, (0, pad_size), value=MASKING_VALUE)
                        padded_outputs.append(out_padded)
                    else:
                        padded_outputs.append(out)
                
                x = torch.stack(padded_outputs)

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
    def __init__(self, embed_dim):
        super().__init__()
        self.mamba = Mamba2(d_model=embed_dim, d_state=128, d_conv=4, expand=2)
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

class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim, patch_size=13, use_pos_enc=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_pos_enc = use_pos_enc
        self.norm = nn.LayerNorm(embed_dim)

        self.time_module = TimeModule(embed_dim, groups=8, patch_size=patch_size)
        self.spectral_module = SpectralModule(embed_dim, patch_size=patch_size)
        self.spatial_module = CoordinatePositionalEncoding(embed_dim)
        if self.use_pos_enc:
            self.trial_temporal_module = TrialTemporalEncoding(embed_dim, patch_size=patch_size)
        self.pad_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))


    def forward(self, x, coords=None):
        # Split into patches
        # Input (B, C, T)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size) # (B, C, T//patch_size, patch_size)
        # Remove PE (last channel)
        if self.use_pos_enc:
            pe = x[:, -1:, :, :]
            x = x[:, :-1, :, :]

        x_time, ch_mask = self.time_module(x)
        # x_spectral = self.spectral_module(x)
        x_total = x_time #+ x_spectral

        x_pos = self.spatial_module(coords, ch_mask)
        x_pos = x_pos.unsqueeze(2)  # (B, C, 1, D)

        if self.use_pos_enc:
            x_trial = self.trial_temporal_module(pe)
        else:
            x_trial = 0

        x_total = x_total + x_pos + x_trial

        pad = self.pad_token.expand_as(x_total)
        x_total = torch.where(ch_mask.unsqueeze(-1), x_total, pad)
        B, C, n, D = x_total.shape
        x_total = x_total.reshape(B, C*n, D)

        return x_total, ch_mask


class TimeModule(nn.Module):
    def __init__(self, embed_dim, groups=8, patch_size=13):
        super().__init__()
        self.embed_dim = embed_dim
        self.groups = groups
        self.patch_size = patch_size

        self.conv = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=1, padding=0)
        self.norm = nn.GroupNorm(num_groups=embed_dim // groups, num_channels=embed_dim)
        self.act = nn.GELU()


    def forward(self, x):
        # (B, C, T//patch_size, patch_size)
        B, C, n, t = x.shape
        ch_mask = (x[:, :, 0, 0] != MASKING_VALUE).unsqueeze(-1).unsqueeze(-1).to(x.device)
        x = x * ch_mask
        x = x.reshape(B*C*n, 1, t)

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        x = x.reshape(B, C, n, -1)
        x = x * ch_mask
        return x, ch_mask.squeeze(-1)
    
class SpectralModule(nn.Module):
    def __init__(self, embed_dim, patch_size=13):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(nn.Linear((patch_size // 2) + 1, self.embed_dim), nn.Dropout(0.1))
    
    def forward(self, x):
        B, C, n, t = x.shape
        x = x.reshape(B*C*n, t)
        spectral = torch.fft.rfft(x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).view(B, C, n, -1)
        spectral_emb = self.proj(spectral)
        return spectral_emb
    

class CoordinatePositionalEncoding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, emb_dim),
            nn.GELU(),
        )
    
    def forward(self, coords, ch_mask):
        # x: (B, C, 3)
        coords = coords * ch_mask
        pe = self.mlp(coords)
        # pe: (B, C, D)
        return pe


class TrialTemporalEncoding(nn.Module):
    def __init__(self, emb_dim, patch_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.proj = nn.Sequential(nn.Linear(patch_size, emb_dim), nn.GELU())
    
    def forward(self, x):
        # x: (B, C, n, D)
        B, C, n, D = x.shape
        x = x.reshape(B, C*n, D)  # (B, D, C*n)
        x = self.proj(x)
        x = x.unsqueeze(1)
        return x


class ClassificationPrep(nn.Module):
    def __init__(self, emb_dim, patch_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        # Original single-step approach
        self.head = nn.ConvTranspose1d(
            in_channels=self.emb_dim, 
            out_channels=self.emb_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size, 
            padding=0
        )
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1),
        )

    def forward(self, x, max_seq_len):
        # x: (B, L, D) where L = C * n_patches
        B, L, D = x.shape
        n_patches = max_seq_len // self.patch_size
        C = L // n_patches

        x = x.view(B, C, n_patches, D)

        # # Mean over channels
        # x = x.mean(dim=1)

        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * C, D, n_patches)

        y = self.head(x)  # (B, D, C*n_patches * patch_size)
        y = self.refine(y)

        T = y.shape[-1]
        y = y.view(B, C, D, T)

        return y


class ClassificationHead(nn.Module):
    def __init__(self, emb_dim, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 3, padding=1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(emb_dim, emb_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(emb_dim // 2, n_classes, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(0)
        y = self.classifier(x)
        y = y.squeeze(0).transpose(1, 0)

        return y