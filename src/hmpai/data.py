from hmpai.utilities import set_seaborn_style
from matplotlib import pyplot as plt
import netCDF4
import xarray as xr
import hmp
import numpy as np
from pathlib import Path
from hmpai.utilities import get_masking_indices_xr
from hmpai.behaviour.sat2 import read_behavioural_info, merge_data_xr


SAT_CLASSES_ACCURACY = [
    "negative",
    "encoding",
    "decision",
    "confirmation",
    "response"
]

SAT_CLASSES_SPEED = ["negative", "encoding", "decision", "response"]


class StageFinder:
    """
    A class used to fit and label a HMP model on epoched EEG data.

    Attributes
    ----------
    epoch_data : xarray.Dataset
        The epoched EEG data.
    verbose : bool
        Whether to print verbose output.
    labels : list of str
        The labels for each stage.
    conditions : list of dict
        The conditions for each stage.
    condition_variable : str
        The variable used to determine the condition.
    condition_method : str
        The method used to check condition equality. ['equals', 'contains']
    duration: int
        The amount of samples that will be included from the start of the stage
    cpus : int
        The number of CPUs to use for fitting the model.
    fit_function : str
        The name of the function used to fit the model.
    fit_args : dict
        The arguments used to fit the model.
    n_comp : int
        The number of principal components to use for transforming the data.
    models : list of hmp.models.hmp
        The HMP models fitted for each condition.
    fits : list of hmp.models.fitted
        The fitted HMP models for each condition.
    hmp_data : list of np.ndarray
        The transformed data for each condition.

    Methods
    -------
    fit_model()
        Fits the HMP model on the dataset.
    label_model()
        Labels the dataset using the fitted HMP model.
    visualize_model(positions)
        Visualizes the HMP model.
    """

    def __init__(
        self,
        epoched_data: Path | str | xr.Dataset,
        labels,
        conditions=[],
        condition_variable="cue",
        condition_method="equal",
        extra_split=None,
        cpus=1,
        fit_function="fit",
        fit_args=dict(),
        verbose=False,
        n_comp=None,
        fits_to_load=[],
        event_width=50,
        split_response=False,
        behaviour_path=None,
        behaviour_fn=read_behavioural_info,
        pca_weights=None,
    ):
        # Check for faulty input
        if len(conditions) > 0:
            if type(labels) is dict:
                assert len(labels) == len(
                    conditions
                ), "Not the same amount of conditions as label+conditions combinations"
            else:
                raise ValueError(
                    'Provide conditions as dict(condition: list(labels)), example: {"AC": SAT1_STAGES_ACCURACY, "SP": SAT1_STAGES_SPEED}'
                )
        else:
            if type(labels) is not list:
                raise ValueError(
                    "Provide labels argument as list of strings, denoting stages"
                )

        # Load required data and set up paths
        if isinstance(epoched_data, Path) or type(epoched_data) is str:
            self.epoch_data = xr.open_dataset(epoched_data)
            # self.epoch_data = xr.load_dataset(epoched_data)
        else:
            self.epoch_data = epoched_data
        if behaviour_path is not None:
            self.behaviour = behaviour_fn(behaviour_path)
            self.epoch_data = merge_data_xr(self.epoch_data, self.behaviour)
        self.n_comp = n_comp

        # Transform data into principal component (PC) space
        # will ask in a pop-up how many components to keep
        # selection depends on data size, choose number at cutoff (90/99%) or at 'elbow' point
        print("Transforming epoched data to principal component (PC) space")
        if "offset_before" in self.epoch_data.attrs:
            # Filter out offset_before data so HMP does not use it when fitting
            # Assumes that 'extra_offset' is also in attributes
            # TODO: extra_offset is dependent on RT?
            # Cut off beginning using offset
            epoch_data_no_offset = self.epoch_data.sel(
                samples=range(
                    self.epoch_data.offset_before, len(self.epoch_data.samples)
                )
            )
            # For each epoch, remove last 'extra_offset' samples by setting data to NaN
            reordered = epoch_data_no_offset.stack({'trial_x_participant': ['participant', 'epochs']}).transpose('trial_x_participant', ...)
            indices = get_masking_indices_xr(reordered.data, search_value=np.nan)
            for i, index in enumerate(indices):
                reordered.data[i,:,index-reordered.extra_offset:index] = np.nan
            epoch_data_no_offset = reordered.unstack().transpose('participant', 'epochs', ...)

            epoch_data_no_offset["samples"] = range(
                0, len(epoch_data_no_offset.samples)
            )
            self.epoch_data_offset = epoch_data_no_offset
            del reordered
            self.hmp_data_offset = hmp.utils.transform_data(
                self.epoch_data_offset, n_comp=self.n_comp, apply_zscore='all', pca_weights=pca_weights
            )
        else:
            self.epoch_data_offset = self.epoch_data
            self.hmp_data_offset = hmp.utils.transform_data(
                self.epoch_data, n_comp=self.n_comp, apply_zscore='all', pca_weights=pca_weights
            )
        # Split on extra variables before fitting
        if isinstance(extra_split, list):
            for split in extra_split:
                var, method, cond = split
                self.hmp_data_offset = hmp.utils.condition_selection(
                    self.hmp_data_offset,
                    None,
                    cond,
                    variable=var,
                    method=method,
                )
        self.verbose = verbose
        self.labels = labels
        self.main_labels = (
            self.labels[max(self.labels, key=lambda k: len(self.labels[k]))]
            if type(labels) is dict
            else self.labels
        )
        self.conditions = conditions
        self.condition_variable = condition_variable
        self.condition_method = condition_method
        self.cpus = cpus
        self.fit_function = fit_function
        self.fit_args = fit_args
        self.models = []
        self.fits = []
        self.hmp_data = []
        self.fits_to_load = fits_to_load
        self.event_width = event_width
        self.split_response = split_response

        if self.verbose:
            print("Epoch data used:")
            print(self.epoch_data)
        # Subset here for easier debugging
        # epoch_data = epoch_data.sel(participant=["0021", "0022", "0023", "0024"])
        # self.epoch_data = self.epoch_data.sel(participant=["0001"])

        return

    # fit_args to optionally give additional arguments when re-fitting in same StageFinder instance
    # extra_split to split on other conditions for just this fit
    def fit_model(self, fit_args: dict = None, extra_split: list = None):
        # Fits HMP model on dataset
        if isinstance(extra_split, list):
            hmp_data_offset = self.hmp_data_offset.copy()
            for split in extra_split:
                var, method, cond = split
                hmp_data_offset = hmp.utils.condition_selection(
                    hmp_data_offset,
                    None,
                    cond,
                    variable=var,
                    method=method,
                )
        else:
            hmp_data_offset = self.hmp_data_offset
        # Keep conditions empty to train HMP model on all data, add conditions to separate them
        # this is useful when conditions cause different stages or stage lengths
        fit_args_tmp = self.fit_args
        if fit_args is not None:
            self.fit_args = self.fit_args | fit_args
        if len(self.conditions) > 0 and extra_split is None:
            for idx, condition in enumerate(self.conditions):
                condition_subset = hmp.utils.condition_selection(
                    hmp_data_offset,
                    None,  # epoch_data deprecated
                    condition,
                    variable=self.condition_variable,
                    method=self.condition_method,
                )
                # Determine amount of expected events from number of supplied labels
                # Subtract 1 since pre-attentive stage does not have a peak
                if self.fit_function == "fit_single":
                    self.fit_args["n_events"] = len(self.labels[condition]) - 1

                # Fit
                if len(self.fits_to_load) > 0:
                    print(f"Loading fitted HMP model for {condition} condition")
                    fit = self.fits_to_load[idx]
                    fit = hmp.utils.load_fit(fit)
                    # Manual transpose after loading https://github.com/GWeindel/hmp/issues/122
                    fit["eventprobs"] = fit.eventprobs.transpose(
                        "trial_x_participant", "samples", "event"
                    )
                    model = hmp.models.hmp(
                        condition_subset,
                        self.epoch_data_offset,
                        cpus=self.cpus,
                        sfreq=self.epoch_data.sfreq,
                        event_width=self.event_width,
                    )
                    self.models.append(model)
                else:
                    print(f"Fitting HMP model for {condition} condition")
                    # Add 
                    fit = self.__fit_model__(condition_subset)
                    self.fit_args = fit_args_tmp

                self.fits.append(fit)
                self.hmp_data.append(condition_subset)
        else:
            print("Fitting HMP model")
            # Determine amount of expected events from number of supplied labels
            # Subtract 1 since pre-attentive stage does not have a peak
            if len(self.fits_to_load) > 0:
                print(f"Loading fitted HMP model for No condition")
                fit = self.fits_to_load[-1]
                fit = hmp.utils.load_fit(fit)
                # Manual transpose after loading https://github.com/GWeindel/hmp/issues/122
                fit["eventprobs"] = fit.eventprobs.transpose(
                    "trial_x_participant", "samples", "event"
                )
                self.fits.append(fit)
                model = hmp.models.hmp(
                    hmp_data_offset,
                    self.epoch_data_offset,
                    cpus=self.cpus,
                    sfreq=self.epoch_data.sfreq,
                    event_width=self.event_width,
                )
                self.hmp_data.append(hmp_data_offset)
                self.models.append(model)
                self.conditions.append("No condition")
            else:
                if self.fit_function == "fit_single":
                    fit_args["n_events"] = len(self.labels) - 1
                fit = self.__fit_model__(hmp_data_offset)
                self.fit_args = fit_args_tmp
                self.fits.append(fit)
                self.hmp_data.append(hmp_data_offset)
                self.conditions.append("No condition")

    def label_model(self, label_fn=None, label_fn_kwargs=None, probabilistic=False):
        model_labels = None
        if label_fn_kwargs is None:
            label_fn_kwargs = {}
        empty = 0 if probabilistic else ""

        for fit, condition in zip(self.fits, self.conditions):
            # Label
            print(f"Labeling dataset for {condition} condition")
            if not probabilistic:
                new_labels = self.__label_model__(
                    fit, label_fn, label_fn_kwargs, condition=condition
                )
            else:
                new_labels = self.__label_model_probabilistic__(
                    fit, condition=condition
                )
            if model_labels is None:
                model_labels = new_labels
            else:
                # Merge new labels with old labels, will always be disjoint sets since an epoch can only be one condition
                model_labels = np.where(model_labels == empty, new_labels, model_labels)

        # Add label information to stage_data: ['', '', 'stage1', 'stage1', 'stage2' ...]
        if not probabilistic:
            # Add '' to front equal to self.epoch_data.offset_before
            stage_data = self.epoch_data.assign(
                labels=(["participant", "epochs", "samples"], model_labels)
            )
        else:
            prob_da = xr.DataArray(
                model_labels,
                dims=("participant", "epochs", "labels", "samples"),
                name="probability",
            )
            # prob_da.expand_dims('labels')
            stage_data = self.epoch_data.assign({"probabilities": prob_da})

        return stage_data

    def visualize_model(self, positions, max_time=None, ax=None, colorbar=True, cond_label=None, model_index=None, figsize=None):
        if figsize is not None:
            figsize = figsize
        else:
            figsize = (12, 3)
        set_seaborn_style()
        if model_index is not None:
            hmp.visu.plot_topo_timecourse(
                self.epoch_data_offset,
                self.fits[model_index],
                positions,
                self.models[model_index],
                times_to_display=(np.mean(self.models[model_index].ends - self.models[model_index].starts)),
                max_time=max_time,
                ylabels={"": [self.conditions[model_index].capitalize()] if cond_label is None else [cond_label]},
                as_time=True,
                estimate_method="max",
                ax=ax,
                event_lines=None,
                vmin=-7e-6,
                vmax=7e-6,
                colorbar=colorbar,
                magnify=2.5,
            )
        else:
            fig, ax = plt.subplots(len(self.fits), 1, figsize=figsize, sharex=True, dpi=300)
            for i, condition in enumerate(
                zip(
                    self.fits,
                    self.models,
                    self.hmp_data,
                    self.conditions,
                )
            ):
                sfreq = self.epoch_data.sfreq
                max_time = (
                    max_time
                    if max_time is not None
                    else (int(max([len(fit.samples) for fit in self.fits]) / 2) / sfreq)
                    * 1000
                )
                ax_i = hmp.visu.plot_topo_timecourse(
                    self.epoch_data_offset,
                    condition[0],
                    positions,
                    condition[1],
                    times_to_display=(np.mean(condition[1].ends - condition[1].starts)),
                    max_time=max_time,
                    ylabels={"": [condition[3].capitalize()]},
                    as_time=True,
                    estimate_method="max",
                    ax=ax[i],
                    event_lines=None,
                    vmin=-7e-6,
                    vmax=7e-6,
                    colorbar=colorbar,
                    magnify=2.5,
                )
                ax[i] = ax_i
            ax[-1].set_xlabel("Time (in ms)")
            fig.supylabel("Condition")
            return fig, ax
        
    def __fit_model__(self, hmp_data):
        # Initialize model
        model = hmp.models.hmp(
            hmp_data,
            self.epoch_data_offset,
            cpus=self.cpus,
            sfreq=self.epoch_data.sfreq,
            event_width=self.event_width,
        )
        self.models.append(model)

        # Using the provided fit_function name, attempt to fit the model
        if hasattr(model, self.fit_function):
            func = getattr(model, self.fit_function)
            try:
                fitted = func(**self.fit_args)
                return fitted
            except Exception as e:
                raise ValueError(
                    f"An error occurred when trying to fit the model with the provided arguments. Error details: {str(e)}"
                )
        else:
            available_methods = [
                method_name
                for method_name in dir(model)
                if callable(getattr(model, method_name))
            ]
            raise ValueError(
                f'Provided fit_function "{self.fit_function}" not found on model instance. Available methods are: {available_methods}'
            )

    def __label_model__(self, model, label_fn, label_fn_kwargs, condition=None):
        n_events = len(model.event)
        if condition == "No condition":
            condition = None
        labels = self.labels if condition is None else self.labels[condition]
        self.negative_class_start_idx = 0

        if len(labels) - 1 != n_events:
            raise ValueError(
                "Amount of labels is not equal to amount of events, adjust labels parameter"
            )

        # Set up output datatypes
        event_locations = model.eventprobs.idxmax(dim="samples").astype(int)

        shape = (
            len(self.epoch_data.participant),
            len(self.epoch_data.epochs),
            len(self.epoch_data.samples),
        )

        # Fill with empty strings since XArray saves np.NaN and other None values as empty strings anyway
        # this avoids discrepancies in data
        labels_array = np.full(shape, fill_value="", dtype=object)
        participants = list(self.epoch_data.participant.values)
        prev_participant = None

        # For every known set of event locations, find the EEG data belonging to that trial (epoch) and participant
        for locations, data in zip(event_locations, model.trial_x_participant):
            data = data.item()
            locations = locations.values
            # Shift locations by one backwards
            locations = locations - 1
            if locations[0] < 0:
                locations[0] = 0

            # Handle participant change
            participant = participants.index(data[0])
            if participant != prev_participant:
                print(f"Processing participant {data[0]}")

            # Skip epoch if bumps are predicted to be at the same time
            unique, counts = np.unique(locations, return_counts=True)
            if np.any(counts > 1):
                continue

            # Skip epoch if bump order is not sorted, can occur if probability mass is greater after the max probability of an earlier bump
            if not np.all(locations[:-1] <= locations[1:]):
                continue

            # TODO: Look into if this changes for SAT2
            # data[1] = epoch number (0...2446)
            # epoch = data[1]
            epochs = list(self.epoch_data.epochs.to_numpy())
            epoch = epochs.index(data[1])

            if epoch > shape[1]:
                print("Epoch number exceeds shape of data, skipping")
                continue

            RT_data = self.epoch_data.sel(participant=data[0], epochs=data[1]).rt.item()
            RT_sample = int(RT_data * self.epoch_data.sfreq)

            prev_participant = participant

            # Calls function that puts the labels corresponding to the given locations in labels_array
            label_fn(
                locations,
                RT_sample,
                participant,
                epoch,
                labels,
                labels_array,
                **label_fn_kwargs,
            )

            self.negative_class_start_idx = (
                self.negative_class_start_idx + 1
                if self.negative_class_start_idx + 1 < len(labels) - 1
                else 0
            )

        return np.copy(labels_array)

    def __label_model_probabilistic__(self, model, condition=None):
        n_events = len(model.event)
        if condition == "No condition":
            condition = None
        labels = self.labels if condition is None else self.labels[condition]
        # Handle case where response_left/response_right are same event but labels is longer
        # if len(labels) - 1 != n_events:
        #     raise ValueError(
        #         "Amount of labels is not equal to amount of events, adjust labels parameter"
        #     )
        shape = list(self.epoch_data.data.shape)
        # Instead of channels, create a dim for label probabilities
        shape[-2] = len(self.main_labels)
        labels_array = np.full(shape, fill_value=0, dtype=np.float32)
        # participants = list(self.epoch_data.participant.values)
        # prev_participant = None

        # (trials, samples, events)
        probs = model.eventprobs.unstack()
        dims = probs.dims
        participants = probs.participant.to_numpy()
        probs = probs.transpose(dims[2], dims[3], dims[1], dims[0])
        epochs = list(self.epoch_data.epochs.to_numpy())
        for participant in np.arange(probs.shape[0]):
            participant_id = participants[participant]
            print(f"Processing participant {participant_id}")
            for trial in probs.trials:
                trial_data = probs.sel(participant=participant_id, trials=trial)
                if not np.any(np.isnan(trial_data.data)):
                    for event in trial_data.event:
                        # Translate from event index (0,1,2,3 for acc, 0,1,2 for spd) to index in labels_array
                        # Assumes labels start with 'negative' (irrelevant for this method of labelling)
                        event_idx = self.main_labels.index(labels[event.item() + 1])
                        event_data = trial_data.sel(event=event).data
                        if "offset_before" in self.epoch_data.attrs:
                            # Left pad using offset
                            event_data = np.pad(
                                event_data,
                                pad_width=((self.epoch_data.offset_before, 0)),
                                mode="constant",
                                constant_values=0,
                            )
                        if "extra_offset" in self.epoch_data.attrs:
                            # Right pad using extra_offset
                            event_data = np.pad(
                                event_data,
                                pad_width=((0, self.epoch_data.extra_offset)),
                                mode="constant",
                                constant_values=0,
                            )
                        # TODO: Bad way to do this, this is true for everything in this condition
                        # Right pad if condition samples is shorter than expected samples
                        if event_data.shape[-1] < labels_array.shape[-1]:
                            event_data = np.pad(
                                event_data,
                                pad_width=(
                                    (0, labels_array.shape[-1] - event_data.shape[-1]),
                                ),
                                mode="constant",
                                constant_values=0,
                            )
                        # Index from premade list of epochs
                        # : dimension (samples) differs across condition
                        epoch = epochs.index(trial.item())
                        if self.split_response and event_idx == 4 and self.epoch_data.sel(participant=participant_id, epochs=epoch).event_name.str.contains('right'):
                            event_idx += 1
                        labels_array[
                            participant, epoch, event_idx, :
                        ] = event_data

        return np.copy(labels_array)