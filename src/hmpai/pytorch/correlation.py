import torch
from scipy.stats import spearmanr, wasserstein_distance
from scipy.special import kl_div
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, jensenshannon


def pearson(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    # Assuming that t2 is hmp
    t1 = t1.detach().to('cpu')
    t2 = t2.detach().to('cpu')
    correlations = []

    for i in range(t1.shape[-1]):
        if i == 0:
            continue
        t1_class = t1[..., i].squeeze()
        t2_class = t2[..., i].squeeze()

        # non_zero = t2_class != 0
        # t1_class = t1_class[non_zero]
        # t2_class = t2_class[non_zero]

        mean1 = t1_class.mean()
        mean2 = t2_class.mean()

        dev1 = t1_class - mean1
        dev2 = t2_class - mean2

        covariance = torch.sum(dev1 * dev2)
        std1 = torch.sqrt(torch.sum(dev1**2))
        std2 = torch.sqrt(torch.sum(dev2**2))

        # Case where all values in t1_class or t2_class are the same (no std defined)
        if std1 == 0 or std2 == 0:
            correlations.append(torch.nan)
        else:
            correlation = covariance / (std1 * std2)
            correlations.append(correlation.item())

    return torch.tensor(correlations).to("cpu")


def spearman(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    # Assuming that t2 is hmp
    t1 = t1.detach().to("cpu")  # Flatten except last dim (labels)
    t2 = t2.detach().to("cpu")

    correlations = []

    for i in range(t1.shape[-1]):
        if i == 0:
            continue
        t1_class = t1[..., i].squeeze()
        t2_class = t2[..., i].squeeze()

        # non_zero = t2_class != 0
        # t1_class = t1_class[non_zero]
        # t2_class = t2_class[non_zero]

        if (t2_class == 0).all():
            correlations.append(torch.nan)
        else:
            spearman_corr = spearmanr(t1_class, t2_class)
            correlations.append(float(spearman_corr[0]))

    return torch.tensor(correlations).to("cpu")


def kldiv(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    # Assuming that t2 is hmp
    # Calculate per-class
    t1 = t1.detach().to("cpu")
    t2 = t2.detach().to("cpu")
    divergences = []

    for i in range(t1.shape[-1]):
        if i == 0:
            continue
        t1_class = t1[..., i].squeeze()
        t2_class = t2[..., i].squeeze()

        # non_zero = t2_class != 0
        # t1_class = t1_class[non_zero]
        # t2_class = t2_class[non_zero]

        if (t2_class == 0).all():
            divergences.append(torch.nan)
            continue

        divergence = kl_div(t1_class, t2_class).mean()
        divergences.append(divergence.item())

    return torch.tensor(divergences)


def dtw(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Calculate DTW distance for each class (label) independently.
    Lower values indicate better alignment.
    """
    t1 = t1.detach().cpu().numpy()
    t2 = t2.detach().cpu().numpy()

    dtw_distances = []

    for i in range(t1.shape[-1]):
        if i == 0:
            continue

        t1_class = t1[..., i].squeeze()
        t2_class = t2[..., i].squeeze()

        # non_zero = t2_class != 0
        # t1_class = t1_class[non_zero]
        # t2_class = t2_class[non_zero]

        if (t2_class == 0).all():
            dtw_distances.append(torch.nan)
            continue
        # Calculate DTW distance for this class
        distance, _ = fastdtw(t1_class, t2_class, dist=2)
        dtw_distances.append(distance)

    return torch.tensor(dtw_distances).to("cpu")


def multiply(t1: torch.Tensor, t2: torch.Tensor):
    t1 = t1.detach().cpu().numpy()
    t2 = t2.detach().cpu().numpy()

    correlations = []

    for i in range(t1.shape[-1]):
        if i == 0:
            continue
        t1_class = t1[..., i].squeeze()
        t2_class = t2[..., i].squeeze()

        correlation = t1_class * t2_class
        correlations.append(correlation.sum())

    return torch.tensor(correlations).to("cpu")


def emd(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    t1 = t1.detach().cpu().numpy()
    t2 = t2.detach().cpu().numpy()

    emd_distances = []

    for i in range(t1.shape[-1]):
        trial_distances = []
        trial_distances.append(0.0)
        if i == 0:
            continue
        t1_class = t1[..., i] # Must be (1, time) if single trial is used
        t2_class = t2[..., i]

        for j in range(t1_class.shape[0]):
            t1_class_trial = t1_class[j, :]
            t2_class_trial = t2_class[j, :]
            if (t2_class_trial == 0).all():
                trial_distances.append(0)
                continue
            trial_distances.append(wasserstein_distance(t1_class_trial, t2_class_trial))
        emd_distances.append(torch.tensor(trial_distances[1:]))
    emd_distances = torch.stack(emd_distances, dim=1)

    return emd_distances


def jsd(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    t1 = t1.detach().cpu().numpy()
    t2 = t2.detach().cpu().numpy()

    jsd_distances = []

    for i in range(t1.shape[-1]):
        if i == 0:
            continue
        t1_class = t1[..., i].squeeze()
        t2_class = t2[..., i].squeeze()

        if (t2_class == 0).all():
            jsd_distances.append(torch.nan)
            continue

        jsd = jensenshannon(t1_class, t2_class)
        jsd_distances.append(jsd.mean())

    return torch.tensor(jsd_distances).to("cpu")
