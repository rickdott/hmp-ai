from torch import nn
import torch
import torch.nn.functional as F
from torch import Tensor
from hmpai.utilities import MASKING_VALUE
import math
from hmpai.pytorch.utilities import DEVICE
from mamba_ssm import Mamba2, Mamba


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
        # distance = torch.matmul(self.pe, self.pe.transpose(1,0))
        # distance_pd = pd.DataFrame(distance.cpu().detach().numpy())
        # distance_pd.to_csv('learn_position_distance.csv')
        return self.dropout(x)


class ClassTokenEmbedding(nn.Module):
    def __init__(self, class_token_dim):
        super().__init__()
        self.class_token = nn.Parameter(
            torch.zeros(1, 1, class_token_dim, device=DEVICE)
        )

    def forward(self, batch_size, sequence_length):
        return self.class_token.expand(batch_size, sequence_length, -1)


class MambaModel(nn.Module):
    def __init__(self, d_model, num_classes, emb_dim):
        super().__init__()
        self.embedding = nn.Linear(d_model, emb_dim)
        self.mamba_encoder = Mamba(d_model=emb_dim, d_state=16, d_conv=4, expand=2)
        self.fc_translation = nn.Linear(emb_dim, d_model)
        self.fc_output = nn.Linear(emb_dim, num_classes)
        self.pretraining = True
        # TESTING

    def set_pretraining(self, is_pretraining):
        self.pretraining = is_pretraining

    def forward(self, x):
        mask = (x == MASKING_VALUE).all(dim=2)
        max_idx = mask.float().argmax(dim=1).max().item()
        mask = mask[:, :max_idx]
        x = x[:, :max_idx, :]
        x = self.embedding(x)
        # x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, feature_dim)

        # transformer_output = self.transformer_encoder(x)
        transformer_output = self.mamba_encoder(x)
        # print(transformer_output.isnan().any())

        # transformer_output = transformer_output.permute(1, 0, 2)

        output = (
            self.fc_translation(transformer_output)
            if self.pretraining
            else self.fc_output(transformer_output)
        )
        # print(output.isnan().any())
        return output


class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, ff_dim, num_heads, num_layers, num_classes, emb_dim):
        super().__init__()
        self.embedding = nn.Linear(d_model, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(emb_dim, num_heads, ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_translation = nn.Linear(emb_dim, d_model)
        self.fc_output = nn.Linear(emb_dim, num_classes)
        self.pretraining = True
        # TESTING

    def set_pretraining(self, is_pretraining):
        self.pretraining = is_pretraining

    def forward(self, x):
        mask = (x == MASKING_VALUE).all(dim=2)
        max_idx = mask.float().argmax(dim=1).max().item()
        mask = mask[:, :max_idx]
        x = x[:, :max_idx, :]
        # print(x.isnan().any())
        x = self.embedding(x)
        # print(x.isnan().any())
        x = self.pos_encoder(x)
        # print(x.isnan().any())
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, feature_dim)

        # transformer_output = self.transformer_encoder(x)
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=mask)
        # print(transformer_output.isnan().any())

        transformer_output = transformer_output.permute(1, 0, 2)

        output = (
            self.fc_translation(transformer_output)
            if self.pretraining
            else self.fc_output(transformer_output)
        )
        # print(output.isnan().any())
        return output


class Seq2SeqTransformerWithChannelConv(nn.Module):
    def __init__(self, d_model, ff_dim, num_heads, num_layers, num_classes, emb_dim):
        super().__init__()
        self.embedding = nn.Linear(d_model, emb_dim)
        self.pos_encoder = PositionalEncoding(128, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(emb_dim, num_heads, ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_translation = nn.Linear(emb_dim, d_model)
        self.fc_output = nn.Linear(emb_dim, num_classes)
        self.pretraining = True
        # TESTING
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(d_model, 1))
        # self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 1), padding='same')

    def set_pretraining(self, is_pretraining):
        self.pretraining = is_pretraining

    def forward(self, x):
        mask = (x == MASKING_VALUE).all(dim=2)
        max_idx = mask.float().argmax(dim=1).max().item()
        mask = mask[:, :max_idx]
        x = x[:, :max_idx, :]
        x = torch.unsqueeze(x, dim=-1)  # (batch_size, seq_len, feature_dim, 1)
        x = x.permute(0, 3, 2, 1)  # (batch_size, 1, feature_dim, seq_len)
        # x = x.permute(0, 2, 1)  # Transformer expects (seq_len, batch_size, feature_dim)
        # (batch_size, feature_dim, seq_len)
        x = self.conv1(x)
        # x = self.conv2(x)
        x = torch.squeeze(x, dim=2)
        # x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, feature_dim)
        # x = x.permute(2, 1, 0)  # Transformer expects (seq_len, batch_size, feature_dim)

        # transformer_output = self.transformer_encoder(x)
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=mask)

        transformer_output = transformer_output.permute(1, 0, 2)

        output = (
            self.fc_translation(transformer_output)
            if self.pretraining
            else self.fc_output(transformer_output)
        )
        return output


class Seq2SeqTransformerWithClassTokens(nn.Module):
    def __init__(self, d_model, ff_dim, num_heads, num_layers, num_classes, emb_dim):
        super().__init__()
        self.class_token_dim = 1
        self.embedding = nn.Linear(d_model, emb_dim)
        self.cls_embedding = ClassTokenEmbedding(self.class_token_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, emb_dim + self.class_token_dim)
        )  # Assuming max seq length of 1000
        # self.pos_encoder = PositionalEncoding(emb_dim + self.class_token_dim, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(
            emb_dim + self.class_token_dim, num_heads, ff_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_translation = nn.Linear(self.class_token_dim, d_model)
        self.fc_output = nn.Linear(self.class_token_dim, num_classes)
        self.pretraining = True
        # TESTING

    def set_pretraining(self, is_pretraining):
        self.pretraining = is_pretraining

    def forward(self, x):
        mask = (x == MASKING_VALUE).all(dim=2)
        max_idx = mask.float().argmax(dim=1).max().item()
        mask = mask[:, :max_idx]
        x = x[:, :max_idx, :]
        x = self.embedding(x)
        class_tokens = self.cls_embedding(x.shape[0], x.shape[1])
        # class_tokens = torch.zeros((x.shape[0], x.shape[1], self.class_token_dim), device=x.device) # (batch_size, seq_len, 1)
        x = torch.cat(
            (class_tokens, x), dim=-1
        )  # Preprend class tokens to last dimension (feature_dim)
        # x = self.pos_encoder(x)
        x = x + self.positional_encoding[:, : x.shape[1], :]
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, feature_dim)

        # transformer_output = self.transformer_encoder(x)
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=mask)

        transformer_output = transformer_output.permute(1, 0, 2)

        output = (
            self.fc_translation(transformer_output[:, :, : self.class_token_dim])
            if self.pretraining
            else self.fc_output(transformer_output[:, :, : self.class_token_dim])
        )
        return output


class TransformerModel(nn.Module):
    def __init__(self, n_features, n_heads, ff_dim, n_layers, n_samples, n_classes):
        super().__init__()
        # self.pos_encoder = tAPE(n_features, max_len=n_samples)
        # self.pos_encoder = LearnablePositionalEncoding(n_features, max_len=n_samples)
        self.pos_encoder = PositionalEncoding(n_features, max_len=n_samples)
        encoder_layers = nn.TransformerEncoderLayer(n_features, n_heads, ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.n_features = n_features
        self.decoder = nn.Linear(n_features, n_classes)
        self.linear = nn.Linear(n_features, n_features)

    def forward(self, x):
        # x = x * math.sqrt(self.n_features)
        mask = (x == MASKING_VALUE).all(dim=2).t()
        max_idx = mask.float().argmax(dim=0).max()
        mask = mask[:max_idx, :]
        x = x[:, :max_idx, :]
        # print(f"Mask == true: {(mask[0,:] == True).any()}")
        # print(mask[0, :])
        x = self.linear(x)
        # print(x.isnan().any())
        pos_enc = self.pos_encoder(x)
        # print(pos_enc.isnan().any())
        # Remove mask if writing summary
        # x = self.transformer_encoder(pos_enc)
        x = self.transformer_encoder(pos_enc, src_key_padding_mask=mask)
        x = torch.nan_to_num(x)
        # print(x.isnan().any())
        inverse_mask = ~mask
        inverse_mask = inverse_mask.float().t().unsqueeze(-1)
        # print(inverse_mask.isnan().any())

        x = x * inverse_mask
        # print(x.isnan().any())

        sum_emb = x.sum(dim=1)
        # print(sum_emb.isnan().any())

        sum_mask = inverse_mask.squeeze(-1).sum(dim=1, keepdim=True)
        # print(sum_mask.isnan().any())
        mean_pooled = (sum_emb / sum_mask).clamp(min=1)
        # print(mean_pooled.isnan().any())

        x = self.decoder(mean_pooled)
        return x


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=250):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float()
#             * (-torch.log(torch.tensor(10000.0)) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).to(DEVICE)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         x = x + self.pe[:, : x.size(1), :].to(DEVICE)
#         return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
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
        # x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        # x = self.maxpool(x)
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


class SAT1Mamba(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        # self.mamba = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.linear = nn.LazyLinear(n_classes)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(128)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = self.norm(x)
        x = x[:, -1, :]
        # x = x.mean(dim=1)
        x = self.linear(x)
        return x


class MambaModel(nn.Module):
    def __init__(self, embed_dim, n_channels, n_classes, n_layers, global_pool=False, dropout=0):
        super().__init__()
        self.blocks = nn.Sequential(*[MambaBlock(embed_dim, dropout) for _ in range(n_layers)])
        self.global_pool = global_pool
        self.linear_in = nn.Linear(n_channels, embed_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, out_channels=128, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.linear_out = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        mask = (x == MASKING_VALUE).all(dim=2).t()
        max_idx = mask.float().argmax(dim=0).max()
        mask = mask[:max_idx, :]
        x = x[:, :max_idx, :]
        # x = x.permute(0, 2, 1)
        # x = self.cnn(x)
        # x = x.permute(0, 2, 1)
        x = self.linear_in(x)
        out = self.blocks(x) if not self.global_pool else torch.mean(self.blocks(x), dim=1)
        out = self.linear_out(out)
        return out

# https://github.com/apapiu/mamba_small_bench
class MambaBlock(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super().__init__()

        self.mamba = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        # self.mamba = Mamba2(d_model=embed_dim, d_state=128, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(self.mamba(x) + x)
        return self.dropout(x)

class SAT1GRU(nn.Module):
    def __init__(self, n_channels, n_classes):
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
