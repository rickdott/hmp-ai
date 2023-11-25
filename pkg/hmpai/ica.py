from mne.io import read_info
import mne
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import xarray as xr
from hmpai.utilities import MASKING_VALUE
import matplotlib.pyplot as plt
import numpy as np


class ICA:
    def __init__(
        self,
        dataset: xr.Dataset,
        ica_path: Path | str = None,
        info_path: Path | str = None,
    ):
        # Calculate ICA based on first dataset entered or load from file
        if ica_path is None:
            if info_path or dataset is None:
                raise ValueError(
                    "If ica_path is not provided, info_path must be provided"
                )
            else:
                # Calculate ICA from dataset
                self.ica = self.calculate_ica(dataset)
                # Re-order dataset using self.ica
                self.info = read_info(info_path)
        else:
            self.ica = mne.preprocessing.read_ica(ica_path)
            # Re-order dataset using self.ica, ica should be calculated from same dataset
            self.info = self.ica.info

    def calculate_ica(self, dataset: xr.Dataset):
        n_comp = len(dataset.channels)
        ica = mne.preprocessing.ICA(n_components=n_comp, random_state=42)

        # Remove singular dimensions
        data = dataset.data.squeeze()
        mask = data != MASKING_VALUE

        # Mask out masking value and reshape to (index, channels)
        data = data[mask].reshape(-1, data.shape[2])
        raw = mne.io.RawArray(data.T, self.info)

        ica.fit(raw)

        return ica

    def reorder_dataset(self, dataset: xr.Dataset):
        # Reorder dataset to match ICA components of original dataset
        corrs, ica2 = self.correlation(dataset)
        x, y = self.match_components(corrs)

        ica2.unmixing_matrix_ = ica2.unmixing_matrix_[y, :]
        ica2.mixing_matrix_ = ica2.mixing_matrix_[:, y]

        # (index, samples, channels) > (index, samples, components)
        for i in range(dataset.data.shape[0]):
            raw_array = mne.io.RawArray(dataset.data[i].T, self.info)
            ica_data = ica2.get_sources(raw_array).get_data().T
            sources = np.where(dataset.data[i] != 999, ica_data, 999)
            dataset.data[i] = sources

        return dataset

    def plot_ica(self, ica):
        ica.plot_components(show=False)
        plt.show()

    def correlation(self, dataset):
        # Calculate correlation between self.ICA components and ICA components calculated from new dataset
        ica2 = self.calculate_ica(dataset)

        corrs = np.empty((self.ica.n_components_, ica2.n_components_))
        for i, comp1 in enumerate(self.ica.get_components().T):
            for j, comp2 in enumerate(ica2.get_components().T):
                corr = np.corrcoef(comp1, comp2)[0, 1]
                corrs[i, j] = abs(corr)

        return corrs, ica2

    def plot_correlation(self, corrs):
        plt.imshow(corrs, cmap="hot")
        plt.xticks(np.arange(corrs.shape[0]))
        plt.xlabel("ICA 2")
        plt.ylabel("ICA 1")
        plt.yticks(np.arange(corrs.shape[1]))
        plt.colorbar()
        plt.show()

    def match_components(self, corrs):
        x, y = linear_sum_assignment(-corrs)
        print("Mean correlation: ", corrs[x, y].mean())

        return x, y
