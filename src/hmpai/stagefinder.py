from hmpai.utilities import set_seaborn_style
from matplotlib import pyplot as plt
import netCDF4
import xarray as xr
import hmp
import numpy as np
from pathlib import Path
from hmpai.utilities import get_masking_indices_xr
from hmpai.behaviour.sat2 import read_behavioural_info, merge_data_xr
from typing import Callable, Type
import pickle
from hmpai.projectors import defaultKeepData
from graphlib import TopologicalSorter


class StageFinder:
    def __init__(
        self,
        epoched_data: Path | str | xr.Dataset,
        conditions: list[str] | None = None,
        models: (
            list[Path | str | hmp.models.base.BaseModel] | None
        ) = None,  # Must be equal length to conditions
        estimates: (
            list[Path | str | xr.DataArray] | None
        ) = None,  # Must be equal length to conditions and models
        offsets_hmp: tuple[float, float] = (0, 0), # Offsets to be used in base_data
        offsets_mamba: tuple[float, float] = (0, 0), # Offsets to be used in data that will be labelled, will be added to offsets_HMP to create final offsets (past HMP)
        preprocessing_kwargs: (
            dict | None
        ) = None,  # Parameters used in `hmp.preprocessing.Standard`, provide pca weights here as 'weights' and # PCA components as 'n_comp'
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
        if len(self.conditions) == 0:
            if len(self.models) > 0:
                self.conditions = ["No condition"]

        self.offsets_hmp, self.offsets_mamba = offsets_hmp, offsets_mamba
        if offsets_hmp is not None:
            if offsets_mamba is not None:
                self.offsets_mamba = tuple(a + b for a, b in zip(offsets_hmp, offsets_mamba))

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

        # Preprocess, split after if necessary per condition
        if len(self.models) == 0 and len(self.estimates) == 0:
            # First preprocess using only offsets_hmp
            first_run_kwargs = self.preprocessing_kwargs.copy()
            first_run_kwargs["offsets"] = offsets_hmp

            self.first_run_kwargs = first_run_kwargs
            self.preprocessed = defaultKeepData(self.epoched_data, **first_run_kwargs)
            self.epoched_data_no_offset = self.preprocessed.data_epoched

            second_run_kwargs = self.preprocessing_kwargs.copy()
            del second_run_kwargs['n_comp']
            second_run_kwargs["offsets"] = offsets_mamba
            full_prep = defaultKeepData(self.epoched_data, weights=self.preprocessed.projector.weights, for_mamba=True, **second_run_kwargs)
            self.epoched_data = full_prep.data_epoched
            del full_prep
            self.second_run_kwargs = second_run_kwargs

    def fit_model(
        self,
        model_class: Type[hmp.models.base.BaseModel],
        model_kwargs: dict = dict(),
        condition_variable: str = "condition",
        condition_method: Callable = np.equal,
        event_width: int = None,
        fit_kwargs: dict = dict(),
    ):
        # Optional if models and estimates were provided, will (re)-fill models & estimates lists
        if model_class is hmp.models.CumulativeMethod:
            if event_width is None:
                raise ValueError("Provide event_width when using CumulativeMethod")
            self.event_properties = hmp.patterns.HalfSine(
                width=event_width
            )
        n_events = None
        if model_class == hmp.models.EventModel:
            if "n_events" not in model_kwargs:
                raise ValueError("Provide n_events in model_kwargs when using EventModel, as a dictionary with conditions as keys if fitting multiple conditions")
            n_events = model_kwargs.pop("n_events")

        if len(self.conditions) == 0:
            if n_events is not None:
                model_kwargs['n_events'] = n_events
                model = model_class(**model_kwargs)
                model_kwargs.pop('n_events')
                pattern_data = self.preprocessed
            else:
                model = model_class(self.event_properties, **model_kwargs)
                pattern_data = hmp.patterndata.PatternData.from_basedata(self.preprocessed, pattern=self.event_properties)

            _, estimates = model.fit_transform(pattern_data, **fit_kwargs)
            self.models.append(model)
            self.estimates.append((estimates, self.epoched_data_no_offset))
            self.conditions.append("No condition")
        else:
            for condition in self.conditions:
                print(f"Fitting model for condition: {condition}")
                preprocessed_subset = self.preprocessed.select_coord(condition, condition_variable, condition_method)
                if n_events is not None:
                    model_kwargs['n_events'] = n_events[condition]
                    model = model_class(**model_kwargs)
                    model_kwargs.pop('n_events')
                    pattern_data = self.preprocessed
                else:
                    model = model_class(self.event_properties, **model_kwargs)
                    pattern_data = hmp.patterndata.PatternData.from_basedata(preprocessed_subset, pattern=self.event_properties)
                
                _, estimates = model.fit_transform(pattern_data, **fit_kwargs)
                self.models.append(model)
                self.estimates.append((estimates, self.epoched_data_no_offset))

    def label_model(
        self, labels: list[str] | dict[str, list[str]], all_data: xr.Dataset = None
    ):  # If multiple conditions, should contain a (condition: list of labels) mapping for every condition
        if type(labels) is dict:
            if len(labels) != len(self.models):
                raise ValueError(
                    'Provide conditions as dict(condition: list(labels)), example: {"AC": ["1", "2", "3", "4"], "SP": ["1", "3", "4"]}'
                )
        else:
            if type(labels) is not list:
                raise ValueError(
                    "If no conditions are provided, labels must be a list of labels"
                )
        all_labels = None
        if (
            len(self.models) == 1
            and len(self.estimates) == 1
            and len(self.conditions) == 0
        ):
            self.conditions.append("No condition")

        if all_data is not None:
            kwargs = self.preprocessing_kwargs.copy()
            del kwargs["n_comp"]
            full_prep = defaultKeepData(all_data, weights=self.preprocessed.projector.weights, **kwargs)

            all_data = full_prep.data_epoched
        data = all_data if all_data is not None else self.epoched_data
        self._add_offset_info(data)
        for i, estimate in enumerate(self.estimates):
            condition = self.conditions[i % len(self.conditions)]
            estimate = estimate[0]
            print(f"Labeling dataset for condition: {condition}")
            model_labels = self._label_model(estimate, condition, labels, data)
            if all_labels is None:
                all_labels = model_labels
            else:
                # Merge new labels with old labels, will always be disjoint since a trial + participant combo can only have one condition
                # 0 is valid for probabilistic labels
                all_labels = np.where(all_labels == 0, model_labels, all_labels)

        prob_da = xr.DataArray(
            all_labels,
            dims=("recording", "epoch", "label", "sample"),
            name="probability",
        )
        if all_data is not None:
            labeled_data = all_data.assign({"probabilities": prob_da})
        else:
            labeled_data = self.epoched_data.assign({"probabilities": prob_da})
        return labeled_data

    def _label_model(self, estimate, condition, labels, data):
        # Get union of all label subsets to use as main labels
        if isinstance(labels, dict):
            def merge_ordered(lists):
                ts = TopologicalSorter()
                for seq in lists:
                    for a, b in zip(seq, seq[1:]):
                        ts.add(b, a)
                    for x in seq:
                        ts.add(x)
                return list(ts.static_order())

            main_labels = merge_ordered(labels.values())
        else:
            main_labels = labels

        if condition == "No condition":
            condition = None
        else:
            labels = labels[condition]

        # Use non-offset data
        shape = list(data.data.shape)
        shape[-2] = len(main_labels)
        labels_array = np.zeros(shape, dtype=np.float32)

        # Vectorized preprocessing of estimates
        probs = estimate.unstack()
        pars = [subj for subj in data.recording.values if subj in probs.recording.values]
        probs = probs.sel(recording=pars)
        dims = probs.dims
        participants = probs.recording.values
        probs = probs.transpose(dims[2], dims[3], dims[1], dims[0])

        # Pre-compute epoch mapping for faster lookups
        epochs_list = list(data.epoch.values)
        epoch_to_idx = {epoch: idx for idx, epoch in enumerate(epochs_list)}

        # Pre-compute label mapping
        label_to_idx = {
            labels[i + 1]: main_labels.index(labels[i + 1])
            for i in range(len(labels) - 1)
        }

        # Vectorized processing - get all valid (non-NaN) trials at once
        probs_data = probs.data  # Convert to numpy for speed
        valid_mask = ~np.isnan(probs_data).any(
            axis=-1
        )  # Shape: (recording, epoch, event)

        # Pre-compute padding parameters
        offset_start = np.rint(self.preprocessing_kwargs.get("offsets_mamba", (0, 0))[0] * data.attrs['sfreq']).astype(int)

        # Maybe this has to be reintroduced later? Dont know if there was a reason for it
        # Seems like right-padding to target_length already handles this
        # extra_offset_end = data.attrs.get("extra_offset", data.attrs.get("extra_offset_after_end", 0))
        target_length = labels_array.shape[-1]

        print(f"Labeling {valid_mask.any(axis=-1).sum()} valid trials for condition {condition}")

        # Vectorized trial processing
        for i, participant in enumerate(participants):
            par_i = np.where(data.recording.values == participant)[0][0]
            participant_data = probs_data[i]  # Shape: (epoch, event, sample)
            participant_valid = valid_mask[i]  # Shape: (epoch, event)

            # Process all valid trials for this participant at once
            valid_epochs, valid_events = np.where(participant_valid)

            if len(valid_epochs) == 0:
                continue

            # Batch process all valid trials
            for epoch_idx, event_idx in zip(valid_epochs, valid_events):
                epoch_val = probs.epoch.values[epoch_idx] # Within participant data/subset, global epoch index
                if epoch_val not in epoch_to_idx:
                    continue  # Can occur when the non-offset data is not filtered out during estimation, but the offset data includes for example a value above the threshold
                mapped_epoch_idx = epoch_to_idx[epoch_val]

                # Get label index (assumes labels start with "negative")
                label_key = labels[event_idx + 1]
                main_label_idx = label_to_idx[label_key]

                # Get event data
                event_data = participant_data[epoch_idx, event_idx]

                # Vectorized padding operations
                if offset_start > 0:
                    event_data = np.pad(
                        event_data,
                        (offset_start, 0),
                        mode="constant",
                        constant_values=0,
                    )

                # if extra_offset_end > 0:
                #     event_data = np.pad(
                #         event_data,
                #         (0, extra_offset_end),
                #         mode="constant",
                #         constant_values=0,
                #     )

                # Right pad to target length if needed
                if len(event_data) < target_length:
                    event_data = np.pad(
                        event_data,
                        (0, target_length - len(event_data)),
                        mode="constant",
                        constant_values=0,
                    )
                elif len(event_data) > target_length:
                    event_data = event_data[:target_length]

                # Assign to output array
                labels_array[par_i, mapped_epoch_idx, main_label_idx, :] = event_data

        return labels_array

    def save_model(self, path):
        # path = path / datetime.now().strftime("%Y%m%d%H%M")
        if not path.exists():
            path.mkdir(parents=True)
        total_path = path / "hmp_fit.pkl"
        pickle.dump((self.models, self.estimates), open(total_path, "wb"))


    def visualize_model(self, positions, max_time=None):
        set_seaborn_style()
        if max_time is None:
            raise ValueError("max_time must be provided for visualization, otherwise x-axes will not be the same")
        fig, ax = plt.subplots(
            len(self.estimates), 1, figsize=(10, 1.5 * len(self.estimates))
        )
        axes_list = []
        
        for i, _ in enumerate(self.estimates):
            cur_ax = ax[i] if len(self.estimates) > 1 else ax
            axes_list.append(cur_ax)
            hmp.visu.plot_topo_timecourse(
                self.estimates[i][1], # Data pointer, note that HMP does not use actual RTs from metadata, but the end of the last dist as RT
                self.estimates[i][0], # Estimates
                positions,
                as_time=True,
                event_lines=False,
                ax=cur_ax,
                max_time=max_time,
                # sensors=True,
                # vmin=-7e-6,
                # vmax=7e-6,
            )
            cur_ax.text(
                0,
                1.12,  # (x, y) in axes coordinates
                f"n = {len(self.estimates[i][0].trial)}",
                transform=cur_ax.transAxes,
                ha="left",
                va="top",
            )
            if len(self.conditions) > 1:
                cur_ax.set_ylabel(f"{self.conditions[i % len(self.conditions)]}")
            else:
                cur_ax.set_ylabel(f"{self.conditions[0]}")
            if i != len(self.estimates) - 1 and len(self.estimates) > 1:
                cur_ax.set_xticklabels([])
            if i == len(self.estimates) - 1:
                cur_ax.set_xlabel("Time (in ms)")
        max_xlim = max(ax_.get_xlim()[1] for ax_ in axes_list)
        for ax_ in axes_list:
            ax_.set_xlim(right=max_xlim)
        return fig, ax

    def estimate(self, data, condition_variable=None, condition_method=None):
        kwargs = self.first_run_kwargs.copy()
        del kwargs["n_comp"]
        preprocessed = defaultKeepData(data, weights=self.preprocessed.projector.weights, **kwargs)
        estim_data = preprocessed.data_epoched

        for i, condition in enumerate(self.conditions):
            print(f"Estimating condition: {condition}")
            if condition != 'No condition':
                preprocessed_subset = preprocessed.select_coord(condition, condition_variable, condition_method)
            else:
                preprocessed_subset = preprocessed.data
            model = self.models[i]

            if type(model) is hmp.models.CumulativeMethod:
                pattern_data = hmp.patterndata.PatternData.from_basedata(preprocessed_subset, pattern=self.event_properties)
            else:
                pattern_data = preprocessed_subset

            lkhs, xr_probs = model.transform(pattern_data)
            self.estimates.append((xr_probs, estim_data))

    def _add_offset_info(self, data):
        # HMP offsets
        offset_start = np.rint(self.offsets_mamba[0] * data.attrs['sfreq']).astype(int)
        offset_end = np.rint(self.offsets_mamba[1] * data.attrs['sfreq']).astype(int)

        data.attrs['offset_start'] = offset_start
        data.attrs['offset_end'] = offset_end

        if self.offsets_mamba != (0, 0):
            offset_after_end_hmp = np.rint(self.offsets_hmp[1] * data.attrs['sfreq']).astype(int)
            data.attrs['extra_offset_end'] = offset_end - offset_after_end_hmp
            data.attrs['offset_end'] = offset_after_end_hmp

            offset_before_start_hmp = np.rint(self.offsets_hmp[0] * data.attrs['sfreq']).astype(int)
            data.attrs['extra_offset_start'] = offset_start - offset_before_start_hmp
            data.attrs['offset_start'] = offset_before_start_hmp
        else:
            data.attrs['extra_offset_end'] = 0
            data.attrs['extra_offset_start'] = 0