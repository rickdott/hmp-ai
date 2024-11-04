import torch
from scipy.stats import spearmanr, wasserstein_distance
import torch.nn.functional as F
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, jensenshannon


def correlate(
    corr_fn: str, t1: torch.Tensor, t2: torch.Tensor, nonzero: bool = False
) -> torch.Tensor:
    # Assuming that t2 is hmp
    t1 = t1.detach().to("cpu")
    t2 = t2.detach().to("cpu")

    correlations = []

    for i in range(t1.shape[-1]):
        trial_distances = []
        trial_distances.append(0.0)
        # Skip negative class
        if i == 0:
            continue

        t1_class = t1[..., i]  # Must be (1, time) if single trial is used
        t2_class = t2[..., i]

        for j in range(t1_class.shape[0]):
            t1_class_trial = t1_class[j, :]
            t2_class_trial = t2_class[j, :]
            if nonzero:
                nonzero_indices = t2_class_trial != 0
                t1_class_trial = t1_class_trial[nonzero_indices]
                t2_class_trial = t2_class_trial[nonzero_indices]
            if (t2_class_trial == 0).all():
                trial_distances.append(0)
                continue
            trial_distances.append(TYPES[corr_fn](t1_class_trial, t2_class_trial))
        # Filter out negative class
        correlations.append(torch.tensor(trial_distances[1:]))
    correlations = torch.stack(correlations, dim=1)

    return correlations


def pearson(t1: torch.Tensor, t2: torch.Tensor) -> float:
    mean1 = t1.mean()
    mean2 = t2.mean()

    dev1 = t1 - mean1
    dev2 = t2 - mean2

    covariance = torch.sum(dev1 * dev2)
    std1 = torch.sqrt(torch.sum(dev1**2))
    std2 = torch.sqrt(torch.sum(dev2**2))

    # Case where all values in t1_class or t2_class are the same (no std defined)
    if std1 == 0 or std2 == 0:
        return torch.nan
    else:
        correlation = covariance / (std1 * std2)
        return correlation.item()


def spearman(t1: torch.Tensor, t2: torch.Tensor) -> float:
    if (t2 == 0).all():
        return torch.nan
    else:
        spearman_corr = spearmanr(t1, t2)
        return float(spearman_corr[0])


def kldiv(t1: torch.Tensor, t2: torch.Tensor, epsilon: float = 1e-10) -> float:
    # t1 = t1 + epsilon
    t2 = t2 + epsilon
    # Normalize to make sure they are valid distributions
    # t1 = t1 / t1.sum()
    t2 = t2 / t2.sum()

    t1 = t1.unsqueeze(0)
    t2 = t2.unsqueeze(0)
    if (t2 == epsilon).all():
        return torch.nan

    divergence = F.kl_div(torch.log(t1), t2, reduction='batchmean')
    return divergence

def kldiv_symmetric(t1: torch.Tensor, t2: torch.Tensor, epsilon: float = 1e-10) -> float:
    # t1 = t1 + epsilon
    t2 = t2 + epsilon
    # Normalize to make sure they are valid distributions
    # t1 = t1 / t1.sum()
    t2 = t2 / t2.sum()

    t1 = t1.unsqueeze(0)
    t2 = t2.unsqueeze(0)
    if (t2 == epsilon).all():
        return torch.nan

    forward_divergence = F.kl_div(torch.log(t1), t2, reduction='batchmean')
    backward_divergence = F.kl_div(torch.log(t2), t1, reduction='batchmean')
    return 0.5 * (forward_divergence + backward_divergence)


def dtw(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """
    Calculate DTW distance for each class (label) independently.
    Lower values indicate better alignment.
    """
    if (t2 == 0).all():
        return torch.nan

    # Calculate DTW distance for this class
    distance, _ = fastdtw(t1, t2, dist=2)
    return distance


def multiply(t1: torch.Tensor, t2: torch.Tensor) -> float:
    correlation = t1 * t2
    return correlation.sum()


def emd(t1: torch.Tensor, t2: torch.Tensor) -> float:
    if (t2 == 0).all():
        return torch.nan
    return wasserstein_distance(t1, t2)


def jsd(t1: torch.Tensor, t2: torch.Tensor) -> float:
    if (t2 == 0).all():
        return torch.nan

    jsd = jensenshannon(t1, t2)
    return jsd.mean()


TYPES = {
    "pearson": pearson,
    "spearman": spearman,
    "kldiv": kldiv,
    "dtw": dtw,
    "multiply": multiply,
    "emd": emd,
    "jsd": jsd,
    "kldiv_symmetric": kldiv_symmetric
}
