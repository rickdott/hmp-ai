import torch
from torch import nn
from mamba_ssm import Mamba2, Mamba
from hmpai.utilities import MASKING_VALUE, get_masking_indices

def base_mamba():
    embed_dim = 64
    out_channels = 128
    base_cnn = nn.Sequential(
        nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=50,
            stride=1,
            padding='same',
        ),
        nn.ReLU(),
    )
    model_kwargs = {
        "embed_dim": embed_dim,
        "mamba_dim": out_channels,
        "n_channels": 19,
        "n_classes": 6, # TODO, get label length other way
        "n_mamba_layers": 5,
        "cnn_module": base_cnn,
        "dropout": 0.1,
    }
    model = ConfigurableMamba(**model_kwargs)
    return model

class ConfigurableMamba(nn.Module):
    def __init__(
        self,
        embed_dim: int, # (b, samples, n_channels) -> (b, samples, embed_dim)
        mamba_dim: int, # (b, samples, mamba_dim) -> (b, samples, mamba_dim), must be equal to output feature space of cnn_module
        n_channels: int,
        n_classes: int,
        n_mamba_layers: int,
        cnn_module: nn.Module = None,
        parallel_cnn_modules: list[nn.Module] = None,
        global_pool: bool = False,
        dropout: float = 0,
        space_cnn_module: nn.Module = None,
        time_cnn_module: nn.Module = None,
    ):
        super().__init__()

        # Define sequence of Mamba blocks
        self.blocks = nn.Sequential(
            *[MambaBlock(mamba_dim, dropout) for _ in range(n_mamba_layers)]
        )

        # Linear embedding from feature space (n_channels), to embed_dim
        self.linear_in = nn.Linear(n_channels, embed_dim)

        self.global_pool = global_pool
        self.cnn_module = cnn_module
        self.space_cnn_module = space_cnn_module
        self.time_cnn_module = time_cnn_module
        self.use_cnn = self.cnn_module is not None
        self.two_stream = self.space_cnn_module is not None and self.time_cnn_module is not None
        self.parallel = parallel_cnn_modules is not None
        if self.parallel:
            self.parallel_cnn_modules = parallel_cnn_modules
        self.linear_translation = nn.Linear(mamba_dim, n_channels)
        self.linear_out = nn.Linear(mamba_dim, n_classes)
        self.pretraining = False

    def forward(self, x):
        # Extract function if multiple mamba modules
        max_index = get_masking_indices(x).max()
        x = x[:, :max_index, :]
        
        x = self.linear_in(x)

        if self.use_cnn:
            x = x.permute(0, 2, 1)
            x = self.cnn_module(x)
            x = x.permute(0, 2, 1)
        if self.parallel:
            x = x.permute(0, 2, 1)
            features = []
            for conv in self.parallel_cnn_modules:
                feature = conv(x)
                feature = torch.nn.functional.adaptive_max_pool1d(feature, x.shape[2])
                features.append(feature)
            x = torch.concat(features, dim=1)
            x = x.permute(0, 2, 1)

        out = (
            self.blocks(x)
            if not self.global_pool
            else torch.mean(self.blocks(x), dim=1)
        )
        out = (
            self.linear_out(out)
            if not self.pretraining
            else self.linear_translation(out)
        )
        return out

class TestMamba(nn.Module):
    def __init__(
        self,
        embed_dim: int, # (b, samples, n_channels) -> (b, samples, embed_dim)
        mamba_dim: int, # (b, samples, mamba_dim) -> (b, samples, mamba_dim), must be equal to output feature space of cnn_module
        n_channels: int,
        n_classes: int,
        n_mamba_layers: int,
        global_pool: bool = False,
        dropout: float = 0,
    ):
        super().__init__()
        mamba_dim = n_channels * 4 * 3

        # Define sequence of Mamba blocks
        self.blocks = nn.Sequential(
            *[MambaBlock(mamba_dim, dropout) for _ in range(n_mamba_layers)]
        )
        # Linear embedding from feature space (n_channels), to embed_dim
        # self.linear_in = nn.Linear(n_channels, embed_dim)

        self.pointconv = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        # self.conv_in = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(n_channels, 1), padding=0)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9, padding="same")
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=27, padding="same")

        self.norm = nn.LayerNorm(mamba_dim)

        self.global_pool = global_pool
        self.linear_translation = nn.Linear(mamba_dim, n_channels)
        self.linear_out = nn.Linear(mamba_dim, n_classes)
        self.pretraining = False

    def forward(self, x):
        # Extract function if multiple mamba modules
        # Do with mask and multiplication?
        max_index = get_masking_indices(x).max()
        # [64, 634, 19]
        # [B, T, C]
        x = x[:, :max_index, :]
        # [64, 634-max_index, 19]
        # [B, T-max_index, 19]
        # Linear with same size to learn relationships between electrodes
        # x = self.linear_in(x)

        # Linear (64 > 64), Conv (64 > 128)
        # 1D point-wise convolution to convert 64 channels into 64 features, including correlation across electrodes
        x = x.permute(0, 2, 1)
        x = self.pointconv(x)
        # Dropout on time dimension to decrease dependence on temporal information
        x = nn.Dropout1d(p=0.4)(x)
        x = nn.functional.silu(x)

        # Three conv layers with gradually increasing kernel sizes to capture temporal relationships at different time scales
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = nn.functional.silu(x)

        x = x.permute(0, 2, 1)
        
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



# https://github.com/apapiu/mamba_small_bench
class MambaBlock(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super().__init__()

        # self.mamba = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.mamba = Mamba2(d_model=embed_dim, d_state=64, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(self.mamba(x) + x)
        return self.dropout(x)
