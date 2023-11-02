import torch
import random
import numpy as np
from torchinfo import summary

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_summary_str(model: torch.nn.Module, input_shape: tuple[int, ...]) -> str:
    # Converts model summary to string, to log to Tensorboard
    stats = str(summary(model, input_size=input_shape))
    stats = stats.replace('\n', '<br/>')
    return str(stats)


def set_global_seed(seed: int) -> None:
    # Sets all (hopefully) random states used, to get more consistent training runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
