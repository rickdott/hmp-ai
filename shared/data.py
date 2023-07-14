import xarray as xr
import hsmm_mvpy as hmp
import numpy as np
from pathlib import Path


def add_stages_to_dataset(
    epoched_data_path,
    output_path,
    labels,
    conditions=[],
    condition_variable="cue",
    cpus=1,
    fit_function="fit",
    fit_args=dict(),
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
    epoch_data = xr.load_dataset(epoched_data_path)
    output_path = Path(output_path)

    # Subset here for easier debugging
    epoch_data = epoch_data.sel(participant=["0021", "0022", "0023", "0024"])

    # Transform data into principal component (PC) space
    # will ask in a pop-up how many components to keep
    # selection depends on data size, choose number at cutoff (90/99%) or at 'elbow' point
    print("Transforming epoched data to principal component (PC) space")
    hmp_data = hmp.utils.transform_data(epoch_data)

    # Keep conditions empty to train HMP model on all data, add conditions to separate them
    # this is useful when conditions cause different stages or stage lengths
    if len(conditions) > 0:
        model_labels = None
        for condition in conditions:
            condition_subset = hmp.utils.condition_selection(
                hmp_data, epoch_data, condition, variable=condition_variable
            )

            # Determine amount of expected events from number of supplied labels
            if fit_function == "fit_single":
                fit_args["n_events"] = len(labels[condition])

            # Fit
            print(f"Fitting HMP model for {condition} condition")
            model = fit_model(
                condition_subset,
                epoch_data,
                cpus=cpus,
                fit_function=fit_function,
                fit_args=fit_args,
            )

            # Label
            print(f"Labeling dataset for {condition} condition")
            new_labels = label_model(model, epoch_data, labels[condition])
            if model_labels is None:
                model_labels = new_labels
            else:
                # Merge new labels with old labels, will always be disjoint sets since an epoch can only be one condition
                model_labels = np.where(
                    model_labels == np.nan, new_labels, model_labels
                )
    else:
        print("Fitting HMP model")
        model = fit_model(hmp_data, epoch_data, cpus=cpus)
        print("Labeling dataset")
        model_labels = label_model(model, epoch_data, labels)

    stage_data = epoch_data.assign_coords(
        labels=(["participant", "epochs", "samples"], model_labels)
    )
    return stage_data


def fit_model(hmp_data, epoch_data, cpus=1, fit_function="fit", fit_args=dict()):
    # Initialize model
    model = hmp.models.hmp(hmp_data, epoch_data, cpus=cpus, sfreq=epoch_data.sfreq)

    # Using the provided fit_function name, attempt to fit the model
    if hasattr(model, fit_function):
        func = getattr(model, fit_function)
        try:
            fitted = func(**fit_args)
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
            f'Provided fit_function "{fit_function}" not found on model instance. Available methods are: {available_methods}'
        )


# TODO: Update function description
# Takes a model and the probabilities of events occuring within the dataset the model was initiated on
# and returns an ndarray of shape samples x time x #electrodes
# length of labels must be equal to amount of events
def label_model(model, eeg_data, labels):
    n_events = len(model.event)
    if len(labels) != n_events:
        raise ValueError(
            "Amount of labels is not equal to amount of events, adjust labels parameter"
        )

    # Set up output datatypes
    event_locations = model.eventprobs.idxmax(dim="samples").astype(int)
    # Remove channels dimension
    shape = list(eeg_data.data.shape)
    shape.pop(2)
    labels_array = np.full(shape, fill_value=np.nan, dtype=object)
    participants = list(eeg_data.participant.values)
    prev_participant = None

    # For every known set of event locations, find the EEG data belonging to that trial (epoch) and participant
    for locations, data in zip(event_locations, model.trial_x_participant):
        data = data.item()
        locations = locations.values

        participant = participants.index(data[0])
        if participant != prev_participant:
            print(f"Processing participant {data[0]}")

        # TODO Maybe not reliable enough, what if electrode 0 (Fp1) is working but others are not
        # Find sample for combination of participant + epoch where the value is null, this is the reaction time sample
        # where the participant pressed the button and the last stage ends
        RT_sample = int(
            eeg_data.sel(participant=data[0], epochs=data[1])
            .isnull()
            .argmax("samples")
            .data[0]
        )
        prev_participant = participant
        epoch = data[1]

        # Set stage label for each stage
        for j, location in enumerate(locations):
            # Slice from known event location n to known event location n + 1
            # unless it is the last event, then slice from known event location n to reaction time
            samples_slice = (
                slice(location, locations[j + 1])
                if j != n_events - 1
                else slice(location, RT_sample - 1)
            )
            labels_array[participant, epoch, samples_slice] = labels[j]

    return labels_array


class DataLoader:
    def __init__():
        pass
