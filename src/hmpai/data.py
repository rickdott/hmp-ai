from collections import defaultdict
import xarray as xr
import hmp
import numpy as np
from pathlib import Path, PosixPath
from hmpai.utilities import MASKING_VALUE, CHANNELS_2D
from tqdm.notebook import tqdm


SAT_CLASSES_ACCURACY = [
    "pre-attentive",
    "encoding",
    "decision",
    "confirmation",
    "response",
]

SAT_CLASSES_SPEED = ["negative", "encoding", "decision", "response"]

SAT1_STAGES_ACCURACY = [
    "pre-attentive",
    "encoding",
    "decision",
    "confirmation",
    "response",
]
SAT1_STAGES_SPEED = ["pre-attentive", "encoding", "decision", "response"]
SAT2_STAGES_ACCURACY = [
    "pre-attentive",
    "encoding",
    "decision",
    "confirmation",
    "response",
]
SAT2_STAGES_SPEED = ["pre-attentive", "encoding", "decision", "response"]
AR_STAGES = [
    "pre-attentive",
    "encoding",
    "familiarity",
    "memory",
    "decision",
    "response",
]

COMMON_STAGES = [
    "pre-attentive",
    "encoding",
    "decision",
    "response",
    "confirmation",
    "familiarity",
    "memory",
]


def add_stage_data_to_unprocessed(
    data_path: str | Path, merge_dataset: xr.Dataset
) -> xr.Dataset:
    """Adds stage information from dataset at data_path to merge_dataset, used to label unprocessed data from processed HMP-labeled data

    Args:
        data_path (str | Path): Path to HMP data set with labels, output from Estimation notebook
        merge_dataset (xr.Dataset): Dataset to add labels to

    Returns:
        xr.Dataset: Dataset with added labels
    """
    # Create variables to put new data in
    stage_data = xr.load_dataset(data_path)
    ratio = round(merge_dataset.sfreq / stage_data.sfreq)
    data_shape = stage_data.labels.shape
    new_samples_length = len(merge_dataset.samples)
    new_shape = (data_shape[0], data_shape[1], new_samples_length)

    expanded_data = np.full(new_shape, "", dtype=object)

    def calculate_start_indices(sequence):
        # Find indices where the processing stage changes
        # Count first index as a change
        start_indices = [0]
        current_element = sequence[0]
        for i, element in enumerate(sequence):
            if element != current_element:
                start_indices.append(i)
                current_element = element
        return start_indices

    # For each epoch, find every sequence of processing stages,
    # lengthen it by a factor of 5, and put it into the new dataset.
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            sequence = stage_data.labels[i, j, :].to_numpy()
            start_indices = calculate_start_indices(sequence)

            for i_indices, index in enumerate(start_indices):
                element = sequence[index]
                if element == "":
                    continue

                new_start_pos = index * ratio
                new_end_pos = start_indices[i_indices + 1] * ratio
                if new_end_pos > new_samples_length:
                    new_end_pos = new_samples_length
                expanded_data[i, j, new_start_pos:new_end_pos] = element

    merge_dataset = merge_dataset.assign(
        labels=(["participant", "epochs", "samples"], expanded_data)
    )
    return merge_dataset


def add_stage_dimension(
    data_path: str | Path, merge_dataset: xr.Dataset = None
) -> xr.Dataset:
    """Adds stage dimension to xr.Dataset without a stage dimension.

    Args:
        data_path (str | Path): Path where data can be found, should be in NetCDF (.nc) format.
        Output of estimation notebook is used for this.
        merge_dataset (xr.Dataset, optional): XArray Dataset the stage information should be added to, labels will be used from data_path. If th

    Returns:
        xr.Dataset: Dataset with added stage dimension.
    """
    data_path = Path(data_path)

    stage_data = xr.load_dataset(data_path)
    # Must convert to numpy since np.where does not work in this case on XArray
    label_data = stage_data.labels.to_numpy()

    segments = []
    merge = False

    print("Finding stage changes")
    # Find every position where a stage change occurs
    changes = np.array(np.where(label_data[:, :, :-1] != label_data[:, :, 1:]))
    if merge_dataset is not None:
        merge = True
        # Extrapolate bump locations from ratio between sampling frequencies
        ratio = round(merge_dataset.sfreq / stage_data.sfreq)
    changes[2] += 1

    print("Starting segmentation")
    change_list = list(zip(changes[0], changes[1], changes[2]))
    grouped_changes = defaultdict(list)
    for participant, epoch, change in change_list:
        grouped_changes[(participant, epoch)].append(change)

    for (participant, epoch), change_indices in tqdm(grouped_changes.items()):
        for i, change_idx in enumerate(change_indices):
            # Determine whether to get slice from beginning of epoch or from last change
            sample_slice = (
                slice(0, change_idx)
                if i == 0
                else slice(change_indices[i - 1], change_idx)
            )
            segment = stage_data.isel(
                participant=[participant],  # List to retain dimension in segment
                epochs=[epoch],  # List to retain dimension in segment
                samples=sample_slice,
            )
            if merge:
                merge_slice = slice(
                    sample_slice.start * ratio, sample_slice.stop * ratio
                )
                merge_segment = merge_dataset.isel(
                    participant=[participant],
                    epochs=[epoch],
                    samples=merge_slice,
                )
            if not np.any(segment.labels != ""):
                continue
            label = label_data[participant, epoch, change_idx - 1]
            segment = merge_segment["data"] if merge else segment["data"]
            segment = segment.expand_dims({"labels": 1}, axis=2).assign_coords(
                labels=[label]
            )
            n_samples = len(segment["samples"])
            segment["samples"] = np.arange(0, n_samples)
            if merge and n_samples / change_idx - change_indices[i - 1] != ratio:
                continue  # Skip segment if it does not match the ratio
            segments.append(segment)

    # Recombine into new segments dimension
    print("Combining segments")
    combined_segments = xr.combine_by_coords(segments)
    return combined_segments


def preprocess(
    dataset: xr.Dataset,
    shuffle: bool = True,
    shape_topological: bool = False,
    sequential: bool = False,
    for_ica: bool = False,
) -> xr.Dataset:
    """Preprocess the dataset based on requirements

    Args:
        dataset (xr.Dataset): Input dataset
        shuffle (bool, optional): Whether to shuffle the dataset at the end. Defaults to True.
        shape_topological (bool, optional): Shape the data 3D. Defaults to False.
        sequential (bool, optional): True if input data is sequential instead of split. Defaults to False.
        for_ica (bool, optional): True if input data is for ICA. Defaults to False.

    Returns:
        xr.Dataset: Preprocessed data.
    """
    # Preprocess data
    # Stack dimensions into one MultiIndex dimension 'index'
    stack_dims = ["epochs"]
    if not sequential:
        #TODO: Find out where this was needed?
        stack_dims.append("labels")
        pass
    if not for_ica:
        stack_dims = ["participant"] + stack_dims
    dataset = dataset.stack({"index": stack_dims})
    # Reorder so index is in front and samples/channels are switched, components is used when preprocessing for ICA
    channel_dim = "channels" if "channels" in dataset.dims else "components1"
    dataset = dataset.transpose("index", "samples", ...)
    # Drop all indices for which all channels & samples are NaN, this happens in cases of
    # measuring error or label does not occur under condition in dataset
    dataset = (
        dataset.dropna("index", how="all", subset=["data"])
        if sequential
        else dataset.dropna("index", how="all", subset=["data"])
    )
    dataset = dataset.fillna(MASKING_VALUE)

    if shape_topological:
        dataset = reshape(dataset)
    if shuffle:
        n = len(dataset.index)
        perm = np.random.permutation(n)
        dataset = dataset.isel(index=perm)

    return dataset


def reshape(dataset: xr.Dataset) -> xr.Dataset:
    """
    Reshape the input dataset to a 4D array with dimensions (index, x, y, samples).

    Parameters:
    -----------
    dataset : xr.Dataset
        Input dataset to be reshaped.

    Returns:
    --------
    xr.Dataset
        Reshaped dataset with dimensions (index, x, y, samples).
    """
    # Create array full of 'empty' values (999)
    sparse_height = CHANNELS_2D.shape[0]
    sparse_width = CHANNELS_2D.shape[1]
    reshaped_data = np.full(
        (
            len(dataset.index),
            sparse_height,
            sparse_width,
            len(dataset.samples),
        ),
        MASKING_VALUE,
        dtype=dataset.data.dtype,
    )

    for x in range(sparse_width):
        for y in range(sparse_height):
            if CHANNELS_2D[y, x] == "NA":
                continue
            # Set slice of reshaped data to be information from channel at position in CHANNELS_2D
            reshaped_data[:, y, x, :] = dataset.sel(channels=CHANNELS_2D[y, x]).data

    # Configure new dataset coordinates and assign the reshaped data
    new_coords = (
        dataset.coords.to_dataset()
        .drop_vars("channels")
        .assign_coords({"x": np.arange(sparse_height), "y": np.arange(sparse_width)})
    )
    dataset = new_coords.assign(data=(("index", "x", "y", "samples"), reshaped_data))
    dataset = dataset.transpose("index", "samples", "x", "y")
    return dataset


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
        cpus=1,
        fit_function="fit",
        fit_args=dict(),
        verbose=False,
        n_comp=4,
        fits_to_load=[],
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
            self.epoch_data = xr.load_dataset(epoched_data)
        else:
            self.epoch_data = epoched_data

        self.verbose = verbose
        self.labels = labels
        self.main_labels = self.labels[max(self.labels, key=lambda k: len(self.labels[k]))]
        self.conditions = conditions
        self.condition_variable = condition_variable
        self.condition_method = condition_method
        self.cpus = cpus
        self.fit_function = fit_function
        self.fit_args = fit_args
        self.n_comp = n_comp
        self.models = []
        self.fits = []
        self.hmp_data = []
        self.fits_to_load = fits_to_load

        if self.verbose:
            print("Epoch data used:")
            print(self.epoch_data)
        # Subset here for easier debugging
        # epoch_data = epoch_data.sel(participant=["0021", "0022", "0023", "0024"])
        # self.epoch_data = self.epoch_data.sel(participant=["0001"])

        return

    def fit_model(self):
        # Fits HMP model on dataset

        # Transform data into principal component (PC) space
        # will ask in a pop-up how many components to keep
        # selection depends on data size, choose number at cutoff (90/99%) or at 'elbow' point
        print("Transforming epoched data to principal component (PC) space")
        hmp_data = hmp.utils.transform_data(self.epoch_data)

        # Keep conditions empty to train HMP model on all data, add conditions to separate them
        # this is useful when conditions cause different stages or stage lengths
        if len(self.conditions) > 0:
            for idx, condition in enumerate(self.conditions):
                condition_subset = hmp.utils.condition_selection(
                    hmp_data,
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
                        hmp_data,
                        self.epoch_data,
                        cpus=self.cpus,
                        sfreq=self.epoch_data.sfreq,
                    )
                    self.models.append(model)
                else:
                    print(f"Fitting HMP model for {condition} condition")
                    fit = self.__fit_model__(condition_subset)

                self.fits.append(fit)
                self.hmp_data.append(condition_subset)
        else:
            print("Fitting HMP model")
            # Determine amount of expected events from number of supplied labels
            # Subtract 1 since pre-attentive stage does not have a peak
            if self.fit_function == "fit_single":
                self.fit_args["n_events"] = len(self.labels) - 1
            fit = self.__fit_model__(hmp_data)
            self.fits.append(fit)
            self.hmp_data.append(hmp_data)
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
                new_labels = self.__label_model_probabilistic__(fit, condition=condition)
            if model_labels is None:
                model_labels = new_labels
            else:
                # Merge new labels with old labels, will always be disjoint sets since an epoch can only be one condition
                model_labels = np.where(model_labels == empty, new_labels, model_labels)

        # Add label information to stage_data: ['', '', 'stage1', 'stage1', 'stage2' ...]
        if not probabilistic:
            stage_data = self.epoch_data.assign(
                labels=(["participant", "epochs", "samples"], model_labels)
            )
        else:
            prob_da = xr.DataArray(model_labels, dims=('participant', 'epochs', 'labels', 'samples'), name='probability')
            # prob_da.expand_dims('labels')
            stage_data = self.epoch_data.assign({'probabilities': prob_da})
            
        return stage_data

    def visualize_model(self, positions):
        for condition in zip(
            self.fits,
            self.models,
            self.hmp_data,
            self.conditions,
        ):
            hmp.visu.plot_topo_timecourse(
                self.epoch_data,
                condition[0],
                positions,
                condition[1],
                times_to_display=np.mean(condition[1].ends - condition[1].starts),
                max_time=int(max([len(fit.samples) for fit in self.fits]) / 2),
                figsize=(10, 1),
                ylabels={"Condition": [condition[3]]},
            )

    def __fit_model__(self, hmp_data):
        # Initialize model
        model = hmp.models.hmp(
            hmp_data, self.epoch_data, cpus=self.cpus, sfreq=self.epoch_data.sfreq
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
        # Weighted probability mean method
        # event_locations = (
        #     np.arange(model.eventprobs.shape[1]) @ model.eventprobs.to_numpy()
        # ).astype(int)

        # Remove channels dimension
        shape = list(self.epoch_data.data.shape)
        shape.pop(2)

        # Fill with empty strings since XArray saves np.NaN and other None values as empty strings anyway
        # this avoids discrepancies in data
        labels_array = np.full(shape, fill_value="", dtype=object)
        participants = list(self.epoch_data.participant.values)
        prev_participant = None

        # Mapping from trial_x_participant epoch numbers to dataset epoch numbers
        if condition is None:
            condition_epochs = self.epoch_data.epochs
        elif self.condition_method == "equal":
            condition_epochs = self.epoch_data.where(
                condition == self.epoch_data[self.condition_variable], drop=True
            ).epochs
        elif self.condition_method == "contains":
            condition_epochs = self.epoch_data.where(
                self.epoch_data[self.condition_variable].str.contains(condition),
                drop=True,
            ).epochs
        else:
            raise ValueError(f"Condition method {self.condition_method} not supported")

        if self.verbose:
            print("Epochs used for current condition (if applicable):")
            print(condition_epochs)

        
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
                # Set up condition_epochs, must be done in loop since for SAT2, particicipant + epoch + condition is not unique, in SAT1 each epoch had the same condition across participatns
                participant_data = self.epoch_data.sel(participant=data[0])
                if condition is None:
                    condition_epochs = self.epoch_data.epochs
                elif self.condition_method == "equal":
                    condition_epochs = participant_data.where(
                        participant_data[self.condition_variable] == condition,
                        drop=True,
                    ).epochs
                elif self.condition_method == "contains":
                    condition_epochs = participant_data.where(
                        participant_data[self.condition_variable].str.contains(
                            condition
                        ),
                        drop=True,
                    ).epochs
                condition_epochs = condition_epochs.to_numpy().tolist()

            # Skip epoch if bumps are predicted to be at the same time
            unique, counts = np.unique(locations, return_counts=True)
            if np.any(counts > 1):
                continue

            # Skip epoch if bump order is not sorted, can occur if probability mass is greater after the max probability of an earlier bump
            if not np.all(locations[:-1] <= locations[1:]):
                continue

            # TODO: Look into if this changes for SAT2
            # epoch = data[1]
            # epoch = condition_epochs.index(data[1])
            # # epoch = int(condition_epochs[data[1]])

            if data[1] > shape[1]:
                print("Epoch number exceeds shape of data, skipping")
                continue

            # TODO Maybe not reliable enough, what if electrode 0 (Fp1) is working but others are not
            # Find first sample from the end for combination of participant + epoch where the value is NaN
            # this is the reaction time sample where the participant pressed the button and stage ends
            RT_data = self.epoch_data.sel(
                participant=data[0],
                epochs=data[1],
                channels=self.epoch_data.channels[0],
            ).data.to_numpy()
            RT_idx_reverse = np.argmax(np.logical_not(np.isnan(RT_data[::-1])))
            RT_sample = (
                len(RT_data) - 1
                if RT_idx_reverse == 0
                else len(RT_data) - RT_idx_reverse
            )

            prev_participant = participant

            # Calls function that puts the labels corresponding to the given locations in labels_array
            label_fn(
                locations,
                RT_sample,
                participant,
                data[1],
                labels,
                labels_array,
                **label_fn_kwargs,
            )
            # Increment the index to start looking for negative classes from by 1 if there are indices left, otherwise reset it
            # With 4 classes: 0, 1, 2, 3, 0,...
            # TODO: Maybe only increment if a negative class has been found at that bump location
            self.negative_class_start_idx = self.negative_class_start_idx + 1 if self.negative_class_start_idx + 1 < len(labels) - 1 else 0

        return np.copy(labels_array)

    def __label_model_probabilistic__(self, model, condition=None):
        n_events = len(model.event)
        if condition == "No condition":
            condition = None
        labels = self.labels if condition is None else self.labels[condition]

        if len(labels) - 1 != n_events:
            raise ValueError(
                "Amount of labels is not equal to amount of events, adjust labels parameter"
            )
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
                        labels_array[participant, trial.item(), event_idx, :] = event_data
        
        return np.copy(labels_array)




    def __label_bump_to_bump__(
        self, locations, RT_sample, participant, epoch, labels, labels_array
    ):
        for i, location in enumerate(locations):
            if i == 0:
                # From stimulus onset to first bump == first operation
                initial_slice = slice(0, location)
                labels_array[participant, epoch, initial_slice] = labels[0]
            if i != len(labels) - 1:
                # Operations in between, slice from event to next event as long as the next event timing is not after reaction time (RT)
                if not locations[i + 1] >= RT_sample:
                    samples_slice = slice(location, locations[i + 1])
                else:
                    continue
            else:
                # Last operation, from last event sample to RT (if the event does not occur after RT)
                if not location >= RT_sample:
                    samples_slice = slice(location, RT_sample)
            if self.verbose:
                print(
                    f"i: {i}, Participant: {participant}, Epoch: {epoch}, Sample range: {samples_slice}, Reaction time sample: {RT_sample}"
                )
            labels_array[participant, epoch, samples_slice] = labels[i + 1]

    def __label_samples_around_bump__(
        self,
        locations,
        RT_sample,
        participant,
        epoch,
        labels,
        labels_array,
        window=(0, 0),  # (samples_before, samples_after)
        get_negative_class=True
    ):
        n = len(locations)
        current_idx = self.negative_class_start_idx
        neg_end_idx = (self.negative_class_start_idx - 1) % n
        negative_class_written = False
        total_window_size = sum(window) * 2 + 1
        if current_idx >= n:
            print(" WHOA!")

        while True:
            if not negative_class_written and get_negative_class:
                # Handle the case where the current index is the first one
                if current_idx == 0 and locations[current_idx] > total_window_size:
                    start_idx = locations[current_idx] - window[0] - sum(window) - 1
                    end_idx = locations[current_idx] - window[0]
                    labels_array[participant, epoch, start_idx:end_idx] = labels[0]
                    negative_class_written = True
                # Handle the case where there is more room between transitions than twice the total window size
                elif current_idx != n - 1 and locations[current_idx + 1] - locations[current_idx] > total_window_size:
                    start_idx = locations[current_idx + 1] - window[0] - sum(window) - 1
                    end_idx = locations[current_idx + 1] - window[0]
                    labels_array[participant, epoch, start_idx:end_idx] = labels[0]
                    negative_class_written = True
                # Handle the case where the current index is the last one
                elif current_idx == n - 1 and RT_sample - locations[current_idx] > total_window_size:
                    start_idx = RT_sample - window[0] - sum(window) - 1
                    end_idx = RT_sample - window[0]
                    labels_array[participant, epoch, start_idx:end_idx] = labels[0]
                    negative_class_written = True

            # Add one since slice(x, x) returns nothing, slice(x, x + 1) returns value at index x
            samples_slice = slice(locations[current_idx] - window[0], locations[current_idx] + window[1] + 1)
            if samples_slice.start < 0:
                samples_slice = slice(0, samples_slice.stop)
            if samples_slice.stop > RT_sample:
                samples_slice = slice(samples_slice.start, RT_sample)
            # Add one since we count a bump as the start of an event, so we label it as the operation the event leads into
            labels_array[participant, epoch, samples_slice] = labels[current_idx + 1]

            if current_idx == neg_end_idx:
                break
            current_idx = (current_idx + 1) % n