from hmpai.utilities import set_seaborn_style
from matplotlib import pyplot as plt
import netCDF4
import xarray as xr
import hmp
import numpy as np
from pathlib import Path
from hmpai.utilities import get_masking_indices_xr
from hmpai.behaviour.sat2 import read_behavioural_info, merge_data_xr
from typing import Type
import pickle
from datetime import datetime


class StageFinder:
    def __init__(
        self,
        epoched_data: Path | str | xr.Dataset,
        conditions: list[str] | None = None,
        models: list[
            Path | str | hmp.models.base.BaseModel
        ] | None = None,  # Must be equal length to conditions
        estimates: list[
            Path | str | xr.DataArray
        ] | None = None,  # Must be equal length to conditions and models
        preprocessing_kwargs: dict | None = None,  # Parameters used in `hmp.preprocessing.Standard`, provide pca weights here as 'weights' and # PCA components as 'n_comp'
        # verbose: bool = False,
        # extra_split: list[tuple[]] = None, # Only used in SAT refit currently, when instantiating a StageFinder on a full dataset, then splitting it when doing fit_model
    ):
        self.conditions = conditions or []
        self.models = models or []
        self.estimates = estimates or []
        self.preprocessing_kwargs = preprocessing_kwargs or {}

        # Check for invalid values
        if len(self.conditions) > 0:
            if len(self.models) > 0 and len(self.models) != len(self.conditions):
                raise ValueError(
                    "If conditions are provided, models must be provided as well, and must be equal length to conditions"
                )
            if len(self.estimates) > 0 and len(self.estimates) != len(self.conditions):
                raise ValueError(
                    "If conditions are provided, estimates must be provided as well, and must be equal length to conditions"
                )

        # Load required data, making sure everything is in memory
        if type(epoched_data) is str or type(epoched_data) is Path:
            self.epoched_data = xr.open_dataset(epoched_data)
        else:
            self.epoched_data = epoched_data

        # replace str or path models with deserialized models
        for i, model in enumerate(self.models):
            if isinstance(model, (str, Path)):
                with open(model, "rb") as f:
                    self.models[i] = pickle.load(f)

        # replace str or path estimates with deserialized estimates
        for i, estimate in enumerate(self.estimates):
            if isinstance(estimate, (str, Path)):
                with open(estimate, "rb") as f:
                    self.estimates[i] = pickle.load(f)

        # Create offset-corrected epoched data if required
        if "offset_before" in self.epoched_data.attrs:
            self.epoched_data_no_offset = self.__remove_extra_offset()
        else:
            self.epoched_data_no_offset = self.epoched_data

        # Preprocess, split after if necessary per condition
        if len(self.models) == 0 and len(self.estimates) == 0:
            self.preprocessed = hmp.preprocessing.Standard(
                self.epoched_data_no_offset, copy=True, **self.preprocessing_kwargs
            )

    def fit_model(
        self,
        model_class: Type[hmp.models.base.BaseModel],
        model_kwargs: dict = dict(),
        condition_variable: str = "condition",
        condition_method: str = "equal",
        event_width: int = 50,
    ):
        # Optional if models and estimates were provided, will (re)-fill models & estimates lists
        self.event_properties = hmp.patterns.HalfSine.create_expected(
            sfreq=self.epoched_data_no_offset.sfreq, width=event_width
        )
        if len(self.conditions) == 0:
            model = model_class(self.event_properties, **model_kwargs)
            trial_data = hmp.trialdata.TrialData.from_preprocessed(
                preprocessed=self.preprocessed,
                pattern=model.pattern.template,
            )
            _, estimates = model.fit_transform(trial_data)
            self.models.append(model)
            self.estimates.append(estimates)
            self.conditions.append("No condition")
        else:
            for condition in self.conditions:
                print(f"Fitting model for condition: {condition}")
                preprocessed_subset = hmp.utils.condition_selection(
                    self.preprocessed.data,
                    condition_string=condition,
                    variable=condition_variable,
                    method=condition_method,
                )
                model = model_class(self.event_properties, **model_kwargs)
                trial_data = hmp.trialdata.TrialData.from_preprocessed(
                    preprocessed=preprocessed_subset,
                    pattern=model.pattern.template,
                )
                _, estimates = model.fit_transform(trial_data)
                self.models.append(model)
                self.estimates.append(estimates)

    def label_model(
        self, labels: list[str] | dict[str, list[str]]
    ):  # If multiple conditions, should contain a (condition: list of labels) mapping for every condition
        if type(labels) is dict:
            if len(labels) != len(self.conditions):
                raise ValueError(
                    'Provide conditions as dict(condition: list(labels)), example: {"AC": ["1", "2", "3", "4"], "SP": ["1", "3", "4"]}'
                )
        else:
            if type(labels) is not list:
                raise ValueError(
                    "If no conditions are provided, labels must be a list of labels"
                )
        all_labels = None
        if len(self.models) == 1 and len(self.estimates) == 1 and len(self.conditions) == 0:
            self.conditions.append("No condition")
            
        for model, estimate, condition in zip(
            self.models, self.estimates, self.conditions
        ):
            print(f"Labeling dataset for condition: {condition}")
            model_labels = self._label_model(model, estimate, condition, labels)
            if all_labels is None:
                all_labels = model_labels
            else:
                # Merge new labels with old labels, will always be disjoint since a trial + participant combo can only have one condition
                # 0 is valid for probabilistic labels
                all_labels = np.where(all_labels == 0, model_labels, all_labels)

        prob_da = xr.DataArray(
            all_labels,
            dims=("participant", "epoch", "label", "sample"),
            name="probability",
        )
        labeled_data = self.epoched_data.assign({"probabilities": prob_da})
        return labeled_data

    def _label_model(self, model, estimate, condition, labels):
        # Get condition with most labels to use as main labels
        # TODO: This assumes that any condition with fewer events consists of a subset of the main condition which is not necessarily true
        main_labels = (
            labels[max(labels, key=lambda k: len(labels[k]))]
            if type(labels) is dict
            else labels
        )
        if condition == "No condition":
            condition = None
        else:
            labels = labels[condition]

        # Use non-offset data
        shape = list(self.epoched_data.data.shape)
        shape[-2] = len(main_labels)

        labels_array = np.zeros(shape, dtype=np.float32)

        probs = estimate.unstack()
        dims = probs.dims
        participants = probs.participant.values
        probs = probs.transpose(dims[2], dims[3], dims[1], dims[0])
        epochs = list(self.epoched_data.epoch.values)

        for i, participant in enumerate(participants):
            print(f"Labeling participant {participant} for condition {condition}")
            for epoch in probs.epoch:
                trial_data = probs.sel(participant=participant, epoch=epoch)
                if not np.any(np.isnan(trial_data.data)):
                    for event in trial_data.event:
                        # Assumes labels starts with "negative"
                        event_idx = main_labels.index(labels[event.item() + 1])
                        event_data = trial_data.sel(event=event).data
                        if "offset_before" in self.epoched_data.attrs:
                            # Left pad using offset
                            event_data = np.pad(
                                event_data,
                                pad_width=((self.epoched_data.offset_before, 0)),
                                mode="constant",
                                constant_values=0,
                            )
                        if "extra_offset" in self.epoched_data.attrs:
                            # Right pad using extra offset
                            event_data = np.pad(
                                event_data,
                                pad_width=((0, self.epoched_data.extra_offset)),
                                mode="constant",
                                constant_values=0,
                            )
                        # Right pad if condition is shorter than longest condition (sample dim)
                        if event_data.shape[-1] < labels_array.shape[-1]:
                            event_data = np.pad(
                                event_data,
                                pad_width=(
                                    (0, labels_array.shape[-1] - event_data.shape[-1])
                                ),
                                mode="constant",
                                constant_values=0,
                            )
                        epoch = epochs.index(epoch)
                        labels_array[i, epoch, event_idx, :] = event_data
        return np.copy(labels_array)


    def save_model(self, path):
        path = path / datetime.now().strftime("%Y%m%d%H%M")
        if not path.exists():
            path.mkdir(parents=True)

        # Save all models and estimates
        for i, _ in enumerate(self.models):
            model_path = path / f"model_{i}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.models[i], f)

            estimate_path = path / f"estimates_{i}.pkl"
            with open(estimate_path, "wb") as f:
                pickle.dump(self.estimates[i], f)

    def visualize_model(self, positions):
        set_seaborn_style()
        fig, ax = plt.subplots(len(self.models), 1, figsize=(10, 1.5 * len(self.models)))
        for i, model in enumerate(self.models):
            cur_ax = ax[i] if len(self.models) > 1 else ax
            hmp.visu.plot_topo_timecourse(
                self.epoched_data_no_offset,
                self.estimates[i],
                positions,
                as_time=True,
                event_lines=False,
                ax=cur_ax,
                # max_time=1000,
                # sensors=True,
            )
            cur_ax.text(
                0, 1.12,  # (x, y) in axes coordinates
                f"n = {len(self.estimates[i].trial)}",
                transform=cur_ax.transAxes,
                ha="left", va="top",
            )
            cur_ax.set_ylabel(f"{self.conditions[i]}")
            if i != len(self.models) - 1 and len(self.models) > 1:
                cur_ax.set_xticklabels([])
            if i == len(self.models) - 1:
                cur_ax.set_xlabel("Time (in ms)")
        return fig, ax

    def __remove_extra_offset(self):
        # Remove offset before stimulus
        epoch_data_no_offset = self.epoched_data.sel(
            sample=range(
                self.epoched_data.offset_before, len(self.epoched_data.sample)
            )
        )

        # Remove offset after RT (finds indices per trial where NaN starts)
        if "extra_offset" in epoch_data_no_offset.attrs:
            reordered = epoch_data_no_offset.stack(
                {"trial_x_participant": ["participant", "epoch"]}
            ).transpose("trial_x_participant", ...)
            indices = get_masking_indices_xr(reordered.data, search_value=np.nan)
            for i, index in enumerate(indices):
                reordered.data[i, :, index - reordered.extra_offset : index] = np.nan
            epoch_data_no_offset = reordered.unstack().transpose(
                "participant", "epoch", ...
            )

        epoch_data_no_offset["sample"] = range(0, len(epoch_data_no_offset.sample))
        return epoch_data_no_offset
