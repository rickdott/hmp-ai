import xarray as xr
import hsmm_mvpy as hmp
import numpy as np
from pathlib import Path
from hmpai.utilities import MASKING_VALUE, CHANNELS_2D


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
    last_change = None
    last_epoch = None

    for participant, epoch, change in zip(changes[0], changes[1], changes[2]):
        if last_change is None:
            last_change = change
        else:
            # Dont take segment ending at one epoch and beginning in the next
            if last_change < change:
                segment = stage_data.isel(
                    participant=[participant],  # List to retain dimension in segment
                    epochs=[epoch],  # List to retain dimension in segment
                    samples=slice(last_change, change),
                )
                if merge:
                    merge_segment = merge_dataset.isel(
                        participant=[participant],
                        epochs=[epoch],
                        samples=slice(last_change * ratio, change * ratio),
                    )
                # Ignore start/end segments containing only empty strings
                if np.any(segment.labels != ""):
                    label = segment.labels[0, 0, 0].item()
                    segment = merge_segment["data"] if merge else segment["data"]
                    segment = segment.expand_dims({"labels": 1}, axis=2).assign_coords(
                        labels=[label]
                    )

                    # Reset samples coordinate so it starts at zero
                    segment["samples"] = np.arange(0, len(segment["samples"]))
                    if (
                        merge
                        and (len(segment["samples"]) / (change - last_change)) != ratio
                    ):
                        continue
                    segments.append(segment)
            last_change = change
        if last_epoch != epoch:
            segment = stage_data.isel(
                participant=[participant],  # List to retain dimension in segment
                epochs=[epoch],  # List to retain dimension in segment
                samples=slice(0, change),
            )
            if merge:
                merge_segment = merge_dataset.isel(
                    participant=[participant],  # List to retain dimension in segment
                    epochs=[epoch],  # List to retain dimension in segment
                    samples=slice(0, change * ratio),
                )
            # Ignore start/end segments containing only empty strings
            if np.any(segment.labels != ""):
                label = segment.labels[0, 0, 0].item()
                segment = merge_segment["data"] if merge else segment["data"]
                segment = segment.expand_dims({"labels": 1}, axis=2).assign_coords(
                    labels=[label]
                )

                # Reset samples coordinate so it starts at zero
                segment["samples"] = np.arange(0, len(segment["samples"]))
                segments.append(segment)
        last_epoch = epoch

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
        stack_dims.append("labels")
    if not for_ica:
        stack_dims = ["participant"] + stack_dims
    dataset = dataset.stack({"index": stack_dims})
    # Reorder so index is in front and samples/channels are switched, components is used when preprocessing for ICA
    channel_dim = "channels" if "channels" in dataset.dims else "components1"
    dataset = dataset.transpose("index", "samples", channel_dim)
    # Drop all indices for which all channels & samples are NaN, this happens in cases of
    # measuring error or label does not occur under condition in dataset
    dataset = (
        dataset.dropna("index", how="all", subset=["data"])
        if sequential
        else dataset.dropna("index", how="all")
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
        epoched_data_path,
        labels,
        conditions=[],
        condition_variable="cue",
        cpus=1,
        fit_function="fit",
        fit_args=dict(),
        verbose=False,
        n_comp=4,
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
        self.epoch_data = xr.load_dataset(epoched_data_path)

        self.verbose = verbose
        self.labels = labels
        self.conditions = conditions
        self.condition_variable = condition_variable
        self.cpus = cpus
        self.fit_function = fit_function
        self.fit_args = fit_args
        self.n_comp = n_comp
        self.models = []
        self.fits = []
        self.hmp_data = []

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
            for condition in self.conditions:
                condition_subset = hmp.utils.condition_selection(
                    hmp_data,
                    self.epoch_data,
                    condition,
                    variable=self.condition_variable,
                    method='contains',
                )

                # Determine amount of expected events from number of supplied labels
                # Subtract 1 since pre-attentive stage does not have a peak
                if self.fit_function == "fit_single":
                    self.fit_args["n_events"] = len(self.labels[condition]) - 1

                # Fit
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

    def label_model(self):
        model_labels = None
        for fit, condition in zip(self.fits, self.conditions):
            # Label
            print(f"Labeling dataset for {condition} condition")
            new_labels = self.__label_model__(fit, condition)
            if model_labels is None:
                model_labels = new_labels
            else:
                # Merge new labels with old labels, will always be disjoint sets since an epoch can only be one condition
                model_labels = np.where(model_labels == "", new_labels, model_labels)

        # Add label information to stage_data: ['', '', 'stage1', 'stage1', 'stage2' ...]
        stage_data = self.epoch_data.assign(
            labels=(["participant", "epochs", "samples"], model_labels)
        )
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

    def __label_model__(self, model, condition=None):
        n_events = len(model.event)
        if condition == "No condition":
            condition = None
        labels = self.labels if condition is None else self.labels[condition]

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
        condition_epochs = (
            self.epoch_data.epochs
            if condition is None
            else self.epoch_data.where(
                self.epoch_data[self.condition_variable] == condition, drop=True
            ).epochs
        )
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
            # Skip epoch if bumps are predicted to be at the same time
            unique, counts = np.unique(locations, return_counts=True)
            if np.any(counts > 1):
                continue
            # Skip epoch if bump order is not sorted, can occur if probability mass is greater after the max probability of an earlier bump
            if not np.all(locations[:-1] <= locations[1:]):
                continue
            epoch = int(condition_epochs[data[1]])
            participant = participants.index(data[0])

            if participant != prev_participant:
                print(f"Processing participant {data[0]}")

            # TODO Maybe not reliable enough, what if electrode 0 (Fp1) is working but others are not
            # Find first sample from the end for combination of participant + epoch where the value is NaN
            # this is the reaction time sample where the participant pressed the button and stage ends
            RT_data = self.epoch_data.sel(
                participant=data[0], epochs=epoch, channels=self.epoch_data.channels[0]
            ).data.to_numpy()
            RT_idx_reverse = np.argmax(np.logical_not(np.isnan(RT_data[::-1])))
            RT_sample = (
                len(RT_data) - 1
                if RT_idx_reverse == 0
                else len(RT_data) - RT_idx_reverse
            )

            prev_participant = participant

            # Set stage label for each stage
            for j, location in enumerate(locations):
                # Record labels for pre-attentive stage (from stimulus onset to first peak)
                if j == 0:
                    labels_array[participant, epoch, slice(0, location)] = labels[0]
                # Slice from known event location n to known event location n + 1
                # unless it is the last event, then slice from known event location n to reaction time
                if j != n_events - 1:
                    if not locations[j + 1] >= RT_sample:
                        samples_slice = slice(location, locations[j + 1])
                    else:
                        # End of stage is later than NaNs begin
                        continue
                else:
                    if not location >= RT_sample:
                        samples_slice = slice(location, RT_sample)
                    else:
                        # NaNs begin before beginning of stage, error in measurement, disregard stage
                        continue
                if self.verbose:
                    print(
                        f"j: {j}, Participant: {participant}, Epoch: {epoch}, Sample range: {samples_slice}, Reaction time sample: {RT_sample}"
                    )
                labels_array[participant, epoch, samples_slice] = labels[j + 1]

        return np.copy(labels_array)
