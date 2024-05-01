from hmpai.data import preprocess
from mne.io import read_info
import mne
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import xarray as xr
from hmpai.utilities import MASKING_VALUE
import matplotlib.pyplot as plt
import numpy as np
import torch


class ICA:
    def __init__(
        self,
        dataset: xr.Dataset,
        ica_path: Path | str = None,
        info_path: Path | str = None,
    ):
        # Calculate ICA based on first dataset entered or load from file
        if ica_path is None:
            if info_path is None or dataset is None:
                raise ValueError(
                    "If ica_path is not provided, info_path must be provided"
                )
            else:
                self.info = read_info(info_path)
                # Calculate ICA from dataset
                # self.ica = self.calculate_ica(dataset)
                # Re-order dataset using self.ica
        else:
            self.ica = mne.preprocessing.read_ica(ica_path)
            # Re-order dataset using self.ica, ica should be calculated from same dataset
            self.info = self.ica.info

    def calculate_ica(self, dataset: xr.Dataset, info: mne.Info = None):
        # Somehow works better when preprocessed?
        dataset = preprocess(
            dataset,
            shuffle=False,
            shape_topological=False,
            sequential=True,
            for_ica=True,
        )
        n_comp = len(dataset.channels)
        ica = mne.preprocessing.ICA(n_components=n_comp, random_state=42)
        info = self.info if info is None else info

        # Only fit ICA on non-NaN data, remove all time samples where at least one channel is NaN
        data = dataset.data.to_numpy().reshape(-1, n_comp)
        mask = (data == MASKING_VALUE).any(axis=1)
        data = data[~mask]

        raw = mne.io.RawArray(data.T, info)
        ica.fit(raw)

        return ica

    def reorder(self, dataset: torch.Tensor, info: mne.Info = None):
        info = self.info if info is None else info

        ica2 = self.reorder_components(dataset, info)
        reordered_data = self.reorder_dataset(dataset, info, ica2)
        return reordered_data

    def reorder_components(self, dataset: torch.Tensor, info: mne.Info = None):
        # If info is None, use self.info
        info = self.info if info is None else info
        # Reorder dataset to match ICA components of original dataset
        corrs, ica2 = self.correlation_new_data(dataset, info)
        x, y = self.match_components(corrs)

        ica2.unmixing_matrix_ = ica2.unmixing_matrix_[y, :]
        ica2.mixing_matrix_ = ica2.mixing_matrix_[:, y]

        return ica2

    def reorder_dataset(
        self,
        dataset: xr.Dataset,
        info: mne.Info = None,
    ):
        """
        Reorders the dataset by performing Independent Component Analysis (ICA) on each participant's data,
        and reordering the components based on their correlation with the mean components.

        Args:
            dataset (xr.Dataset): The dataset containing the participant data.
            info (mne.Info, optional): The MNE info object containing information about the data. Defaults to None.

        Returns:
            xr.Dataset: The reordered dataset with additional arrays for unmixing matrices, mixing matrices, and components.
        """
        print("Reordering dataset")
        # If info is None, use self.info
        info = self.info if info is None else info

        # Assume n_components == n_channels
        # (participant, components, components)
        mixing_matrices = np.zeros(
            (len(dataset.participant), len(dataset.channels), len(dataset.channels))
        )
        # (participant, components, components)
        unmixing_matrices = np.zeros(
            (len(dataset.participant), len(dataset.channels), len(dataset.channels))
        )
        # (participant, channels, components)
        components = np.full(
            (len(dataset.participant), len(dataset.channels), len(dataset.channels)),
            np.nan,
        )

        # For every participant, calculate ICA, if not the first participant, match with mean ICA. When a new components is added, its sign is flipped if it is negatively correlated with the mean component.
        for participant in range(len(dataset.participant)):
            print(f"Participant: {participant}")
            p_data = dataset.isel(participant=participant)
            p_ica = self.calculate_ica(p_data, info)
            p_comp = p_ica.get_components()
            mean_comp = np.nanmean(components, axis=0)

            if not np.isnan(mean_comp).any():
                corrs = self.correlation(p_comp, mean_comp)
                x_comps, y_comps = self.match_components(corrs)
                for x, y in zip(x_comps, y_comps):
                    if corrs[x, y] < 0:
                        p_comp[:, x] = -p_comp[:, x]
                p_ica.unmixing_matrix_ = p_ica.unmixing_matrix_[y_comps, :]
                p_ica.mixing_matrix_ = p_ica.mixing_matrix_[:, y_comps]

            unmixing_matrices[participant, :, :] = p_ica.unmixing_matrix_
            mixing_matrices[participant, :, :] = p_ica.mixing_matrix_
            components[participant, :, :] = p_comp

            for epoch in range(len(dataset.epochs)):
                epoch_data = p_data.data[epoch, :, :]
                epoch_data = self.reorder_slice(epoch_data, p_ica, info)
                dataset.data[participant, epoch, :, :] = epoch_data

        # Add unmixing matrices, mixing matrices, and components to dataset to allow for re-calculating original data
        dataset["unmixing_matrices"] = xr.DataArray(
            data=unmixing_matrices,
            dims=["participant", "components1", "components2"],
            coords={
                "participant": dataset.participant,
                "components1": np.arange(p_ica.n_components),
                "components2": np.arange(p_ica.n_components),
            },
        )
        dataset["mixing_matrices"] = xr.DataArray(
            data=mixing_matrices,
            dims=["participant", "components1", "components2"],
            coords={
                "participant": dataset.participant,
                "components1": np.arange(p_ica.n_components),
                "components2": np.arange(p_ica.n_components),
            },
        )
        dataset["components"] = xr.DataArray(
            data=components,
            dims=["participant", "components1", "components2"],
            coords={
                "participant": dataset.participant,
                "components1": np.arange(p_ica.n_components),
                "components2": np.arange(p_ica.n_components),
            },
        )

        # Re-coordinate data variable to components1 dimension
        dataset["data"] = xr.DataArray(
            data=dataset.data,
            dims=["participant", "epochs", "components1", "samples"],
            coords={
                "participant": dataset.participant,
                "epochs": dataset.epochs,
                "components1": dataset.components1,
                "samples": dataset.samples,
            },
        )

        return dataset

    def plot_ica(self):
        self.ica.plot_components(show=False)
        plt.show()

    def save(self, path: Path | str):
        self.ica.save(path)

    def correlation(self, components1: np.array, components2: np.array):
        # Input two ndarrays of shape (channels, components)
        corrs = np.empty((components1.shape[1], components2.shape[1]))
        for i, comp1 in enumerate(components1.T):
            for j, comp2 in enumerate(components2.T):
                corr = np.corrcoef(comp1, comp2)[0, 1]
                corrs[i, j] = corr

        return corrs

    def correlation_new_data(self, dataset: xr.Dataset, info: mne.Info):
        # Calculate correlation between self.ICA components and ICA components calculated from new dataset
        ica2 = self.calculate_ica(dataset, info)

        corrs = np.empty((self.ica.n_components_, ica2.n_components_))
        for i, comp1 in enumerate(self.ica.get_components().T):
            for j, comp2 in enumerate(ica2.get_components().T):
                corr = np.corrcoef(comp1, comp2)[0, 1]
                corrs[i, j] = corr

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
        corrs = abs(corrs)
        x, y = linear_sum_assignment(-corrs)
        print("Mean correlation: ", corrs[x, y].mean())

        return x, y

    def reorder_slice(
        self, slice: xr.DataArray, ica: mne.preprocessing.ICA, info: mne.Info
    ):
        # Takes in slice of shape (channels, samples) and reorders it according to the given ICA
        raw_array = mne.io.RawArray(slice, info, verbose=False)
        ica_data = ica.get_sources(raw_array).get_data()
        # slice = torch.where(slice != 999, torch.tensor(ica_data), torch.tensor(999))
        slice = np.where(slice != MASKING_VALUE, ica_data, MASKING_VALUE)

        return slice
