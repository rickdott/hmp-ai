from mne.viz import plot_topomap
from mne import Info
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from shared.data import SAT1_STAGES_ACCURACY, preprocess
import innvestigate
from shared.generators import SAT1DataGenerator


def add_analysis(
    dataset: xr.Dataset, analyzer: innvestigate.analyzer.base.AnalyzerBase
) -> xr.Dataset:
    test_set = preprocess(dataset)
    test_set = test_set.assign(
        analysis=(("index", "samples", "channels"), np.zeros_like(test_set.data))
    )
    test_gen = SAT1DataGenerator(test_set, do_preprocessing=False)

    for i, batch in enumerate(test_gen):
        batch_analysis = analyzer.analyze(np.expand_dims(batch[0].data, axis=3))
        test_set.analysis[
            i * len(batch[1]) : (i + 1) * len(batch[1]), :, :
        ] = np.squeeze(batch_analysis)
    # TODO: Last few samples have all 0 as analysis since batch is not created for them
    return test_set


def plot_max_activation_per_label(dataset: xr.Dataset, positions: Info) -> None:
    f, ax = plt.subplots(nrows=3, ncols=len(SAT1_STAGES_ACCURACY), figsize=(6, 3))

    for i, label in enumerate(SAT1_STAGES_ACCURACY):
        # Get subset for label
        subset = dataset.sel(labels=label)
        n = len(subset.index)
        # Get maximum activation averaged over channels
        # TODO: Mean channels or sum channels? Maybe something else, highest n values?
        max_activation = subset.mean("channels").argmax("samples").analysis
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
        plot_topomap(
            mean_max_samples.analysis,
            positions,
            axes=ax[1, i],
            show=False,
            cmap="bwr",
            vlim=(np.min(mean_max_samples.analysis), np.max(mean_max_samples.analysis)),
            sensors=False,
            contours=6,
        )

        # Combined
        combined = mean_max_samples.data * mean_max_samples.analysis
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


def plot_single_trial_activation(sample: xr.Dataset, positions: Info) -> None:
    nan_index = np.isnan(sample.analysis.where(sample.analysis != 0)).argmax(
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
            vlim=(np.min(sample.data[0:nan_index, :]), np.max(sample.data[0:nan_index, :])),
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
            vlim=(np.min(sample.analysis), np.max(sample.analysis)),
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
