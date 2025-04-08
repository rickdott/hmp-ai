import pandas as pd
import torch
import random
import numpy as np
from torchinfo import summary
from pathlib import Path
from hmpai.utilities import get_trial_start_end

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_summary_str(model: torch.nn.Module, input_shape: tuple[int, ...]) -> str:
    # Converts model summary to string, to log to Tensorboard
    stats = str(summary(model, input_size=input_shape))
    stats = stats.replace("\n", "<br/>")
    return str(stats)


def set_global_seed(seed: int) -> None:
    # Sets all (hopefully) random states used, to get more consistent training runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_model(
    path: Path,
    epoch: int,
    model_state_dict: dict,
    optimizer_state_dict: dict,
    loss: torch.nn.modules.loss._Loss,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "loss": loss,
        },
        path,
    )


def load_model(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def save_tensor(tensor: torch.Tensor, filename: str) -> None:
    if type(tensor) != np.ndarray:
        np_tensor = tensor.squeeze().numpy()
    else:
        np_tensor = np.squeeze(tensor)
    df_tensor = pd.DataFrame(np_tensor)
    df_tensor.to_csv(filename, index=False)

def add_relative_positional_encoding(data):
    # Data is tuple (data, labels)
    data, probabilities = data
    start, end = get_trial_start_end(probabilities)
    length = data.shape[0]

    encoding_feature = torch.zeros(length)

    pe = torch.arange(length - start).float()
    pe /= end - start
    encoding_feature[start:] = pe.clamp(max=1)
    encoding_feature = encoding_feature.unsqueeze(-1)
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(encoding_feature)
    # plt.vlines([start, end], ymin=0, ymax=1, linestyles="--")
    # plt.show()
    data = torch.cat([data, encoding_feature], dim=1)
    return data, probabilities