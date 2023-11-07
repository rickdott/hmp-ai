from mne.viz import plot_topomap
from mne import Info
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from hmpai.data import SAT1_STAGES_ACCURACY, preprocess
import captum
from hmpai.pytorch.generators import SAT1Dataset
from hmpai.pytorch.utilities import DEVICE
from torch.utils.data import DataLoader
import torch
from hmpai.utilities import MASKING_VALUE
from tqdm.notebook import tqdm


def add_attribution(
    dataset: xr.Dataset, analyzer: captum.attr.Attribution, model: torch.nn.Module
) -> xr.Dataset:
    """
    Analyzes the given dataset using the provided attribution method and model, and adds the analysis
    results to the dataset.

    Args:
        dataset (xr.Dataset): The dataset to analyze.
        analyzer (captum.attr.Attribution): The attribution method to use for analysis.
        model (torch.nn.Module): The model to use for analysis.

    Returns:
        xr.Dataset: The analyzed dataset with the analysis results added.
    """
    test_set = preprocess(dataset)
    test_dataset = SAT1Dataset(test_set, do_preprocessing=False)
    test_set = test_set.assign(
        analysis=(("index", "samples", "channels"), np.zeros_like(test_set.data))
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Batch-wise analyzing of data and adding analysis to the dataset
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # print(f"Batch: {i + 1}/{batches}")
        batch_data = batch[0].to(DEVICE)
        baselines = torch.clone(batch_data)
        mask = baselines != MASKING_VALUE
        baselines[mask] = 0
        baselines = baselines.to(DEVICE)
        target = torch.argmax(model(batch_data), dim=1)
        batch_analysis = analyzer.attribute(
            batch_data,
            baselines=baselines,
            n_steps=100,
            method="riemann_trapezoid",
            # Change to batch[1] if true values should be used instead of model predictions
            target=target,
            internal_batch_size=512,
        )
        test_set.analysis[i * len(batch[1]) : (i + 1) * len(batch[1])] = torch.squeeze(
            batch_analysis.cpu()
        )

    return test_set


def plot_max_activation_per_label(dataset: xr.Dataset, positions: Info) -> None:
    """
    Plots the maximum activation per label for a given dataset.

    Args:
        dataset (xr.Dataset): The dataset to plot.
        positions (Info): The positions of the sensors.

    Returns:
        None
    """
    f, ax = plt.subplots(nrows=3, ncols=len(SAT1_STAGES_ACCURACY), figsize=(6, 3))

    for i, label in enumerate(SAT1_STAGES_ACCURACY):
        # Get subset for label
        subset = dataset.sel(labels=label)
        n = len(subset.index)
        # Get maximum activation averaged over channels
        # TODO: Mean channels or sum channels? Maybe something else, highest n values?
        max_activation = abs(subset.analysis).sum("channels").argmax("samples")
        max_samples = subset.sel(samples=max_activation)
        mean_max_samples = max_samples.mean("index")

        ax[0, i].set_title(f"{label}\n(n={n})", fontsize=10)
        # Row titles
        if i == 0:
            ax[0, i].text(-0.35, 0, "EEG\nActivity", va="center", ha="left")
            ax[1, i].text(-0.35, 0, "Model\nAttention", va="center", ha="left")
            ax[2, i].text(-0.35, 0, "Combined", va="center", ha="left")

        # Raw EEG Activation
        plot_topomap(
            mean_max_samples.data,
            positions,
            axes=ax[0, i],
            show=False,
            cmap="Spectral_r",
            vlim=(np.min(mean_max_samples.data), np.max(mean_max_samples.data)),
            sensors=False,
            contours=6,
        )

        # Model activity
        abs_mean_max_analysis = mean_max_samples.analysis
        plot_topomap(
            abs_mean_max_analysis,
            positions,
            axes=ax[1, i],
            show=False,
            cmap="bwr",
            vlim=(np.min(abs_mean_max_analysis), np.max(abs_mean_max_analysis)),
            sensors=False,
            contours=6,
        )

        # Combined
        combined = mean_max_samples.data * abs(abs_mean_max_analysis)
        plot_topomap(
            combined,
            positions,
            axes=ax[2, i],
            show=False,
            cmap="Spectral_r",
            vlim=(np.min(combined), np.max(combined)),
            sensors=False,
            contours=6,
        )
    plt.tight_layout()
    plt.show()


def plot_mean_activation_per_label(dataset: xr.Dataset, positions: Info) -> None:
    """
    Plots the mean activation per label.

    Parameters:
    -----------
    dataset : xr.Dataset
        The dataset containing the activations.
    positions : dict
        The positions of the sensors.

    Returns:
    --------
    None
    """
    f, ax = plt.subplots(nrows=1, ncols=len(SAT1_STAGES_ACCURACY), figsize=(6, 3))

    for i, label in enumerate(SAT1_STAGES_ACCURACY):
        subset = dataset.sel(labels=label)
        n = len(subset.index)

        mean_activation = subset.mean(["index", "samples"]).data
        ax[i].set_title(f"{label}\n(n={n})", fontsize=10)
        plot_topomap(
            mean_activation,
            positions,
            axes=ax[i],
            show=False,
            cmap="Spectral_r",
            vlim=(np.min(mean_activation), np.max(mean_activation)),
            sensors=False,
            contours=6,
        )

    plt.tight_layout()
    plt.show()


def plot_single_trial_activation(sample: xr.Dataset, positions: Info) -> None:
    """
    Plots the raw EEG activation, model attention, and combined activation for a single trial.

    Args:
        sample (xr.Dataset): The EEG data for a single trial.
        positions (Info): The positions of the EEG sensors.

    Returns:
        None
    """
    nan_index = np.isnan(sample.data.where(sample.data != 999)).argmax(
        dim=["samples", "channels"]
    )
    nan_index = nan_index["samples"].item()
    if nan_index == 0:
        return

    f, ax = plt.subplots(nrows=3, ncols=nan_index, figsize=(nan_index, 3))
    combined = sample.data[0:nan_index, :] * sample.analysis[0:nan_index, :]
    for i in range(nan_index):
        # Raw EEG Activation
        ax[0, i].set_title(f"Sample {i}", fontsize=10)
        # Row titles
        if i == 0:
            ax[0, i].text(-0.35, 0, "EEG\nActivity", va="center", ha="left")
            ax[1, i].text(-0.35, 0, "Model\nAttention", va="center", ha="left")
            ax[2, i].text(-0.35, 0, "Combined", va="center", ha="left")
        plot_topomap(
            sample.data[i, :],
            positions,
            axes=ax[0, i],
            show=False,
            cmap="Spectral_r",
            vlim=(
                np.min(sample.data[0:nan_index, :]),
                np.max(sample.data[0:nan_index, :]),
            ),
            sensors=False,
            contours=6,
        )

        # Model activity
        plot_topomap(
            sample.analysis[i, :],
            positions,
            axes=ax[1, i],
            show=False,
            cmap="bwr",
            vlim=(
                np.min(sample.analysis[0:nan_index, :]),
                np.max(sample.analysis[0:nan_index, :]),
            ),
            sensors=False,
            contours=6,
        )

        # Combined
        plot_topomap(
            combined[i, :],
            positions,
            axes=ax[2, i],
            show=False,
            cmap="Spectral_r",
            vlim=(np.min(combined), np.max(combined)),
            sensors=False,
            contours=6,
        )
    plt.tight_layout()
    plt.show()


def plot_model_attention_over_stage_duration(dataset: xr.Dataset) -> None:
    """
    Plots the model attention over stage duration for the given dataset.

    Parameters:
    dataset (xr.Dataset): The dataset containing the data to be plotted.

    Returns:
    None
    """
    f, ax = plt.subplots(
        nrows=1, ncols=len(SAT1_STAGES_ACCURACY), figsize=(12, 3), sharey=True, sharex=True
    )

    time_points = np.linspace(0, 100, 100)
    for i, label in enumerate(SAT1_STAGES_ACCURACY):
        subset = dataset.sel(labels=label)
        nan_indices = np.isnan(subset.data.where(subset.data != MASKING_VALUE)).argmax(
            dim=["samples", "channels"]
        )
        interpolated = []
        for sample, nan_index in zip(subset.analysis, nan_indices["samples"]):
            sequence = sample.mean(dim="channels")[0 : nan_index.item()]
            if len(sequence) == 0:
                continue
            origin_time_points = np.linspace(0, 100, num=len(sequence))
            interpolated_sequence = np.interp(time_points, origin_time_points, sequence)
            interpolated.append(abs(interpolated_sequence))
        ax[i].plot(np.mean(interpolated, axis=0))
        ax[i].set_title(f"{label}", fontsize=10)
        ax[i].set_yticks(np.arange(0.0, 0.1, 0.02))
    # ax[2].text(0, -0.005, 'Linear interpolation\nof stage length', va='bottom', ha='center')
    ax[0].set_ylabel("Model Attention")
    ax[2].set_xlabel("Stage duration (%)")
    plt.tight_layout()
    plt.show()
