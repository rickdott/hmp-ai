import pandas as pd
import torch
import random
import numpy as np
from torchinfo import summary
from pathlib import Path

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
    return torch.load(path)


def save_tensor(tensor: torch.Tensor, filename: str) -> None:
    if type(tensor) != np.ndarray:
        np_tensor = tensor.squeeze().numpy()
    else:
        np_tensor = np.squeeze(tensor)
    df_tensor = pd.DataFrame(np_tensor)
    df_tensor.to_csv(filename, index=False)