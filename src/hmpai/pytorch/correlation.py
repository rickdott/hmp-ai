import torch
from scipy.stats import spearmanr
from scipy.special import kl_div
from hmpai.pytorch.training import DEVICE


def pearson(t1: torch.Tensor, t2: torch.Tensor) -> float:
    # Assuming that t2 is hmp
    t1 = t1.flatten().to(DEVICE)
    t2 = t2.flatten().to(DEVICE)

    mean1 = t1.mean()
    mean2 = t2.mean()

    dev1 = t1 - mean1
    dev2 = t2 - mean2

    covariance = torch.sum(dev1 * dev2)

    std1 = torch.sqrt(torch.sum(dev1 ** 2))
    std2 = torch.sqrt(torch.sum(dev2 ** 2))

    # Case where all values in t1 or t2 are the same (no std defined)
    if std1 == 0 or std2 == 0:
        return torch.nan

    correlation = covariance / (std1 * std2)

    return correlation.item()

def spearman(t1: torch.Tensor, t2: torch.Tensor) -> float:
    # Assuming that t2 is hmp
    t1 = t1.flatten().to('cpu')
    t2 = t2.flatten().to('cpu')

    non_zero = t2 != 0

    spearman_corr = spearmanr(t1[non_zero], t2[non_zero])

    return float(spearman_corr[0])


def kldiv(t1: torch.Tensor, t2: torch.Tensor) -> list[float]:
    # Assuming that t2 is hmp
    # Calculate per-class
    t1 = t1.detach().to('cpu')
    t2 = t2.detach().to('cpu')
    divergences = []

    for i in range(t1.shape[-1]):
        divergence = kl_div(t1[..., i], t2[..., i]).mean()
        divergences.append(divergence)

    return divergences