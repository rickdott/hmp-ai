import xarray as xr
import hsmm_mvpy as hmp
import numpy as np

SAT1_STAGES_ACCURACY = ["encoding", "decision", "confirmation", "response"]
SAT1_STAGES_SPEED = ["encoding", "decision", "response"]
SAT2_STAGES_ACCURACY = ["encoding", "decision", "confirmation", "response"]
SAT2_STAGES_SPEED = ["encoding", "decision", "response"]
AR_STAGES = ["encoding", "familiarity", "memory", "decision", "response"]


class StageFinder:
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

        if self.verbose:
            print("Epoch data used:")
            print(self.epoch_data)
        # Subset here for easier debugging
        # epoch_data = epoch_data.sel(participant=["0021", "0022", "0023", "0024"])
        # self.epoch_data = self.epoch_data.sel(participant=["0001"])

        return

    def add_stages_to_dataset(self):
        # Transform data into principal component (PC) space
        # will ask in a pop-up how many components to keep
        # selection depends on data size, choose number at cutoff (90/99%) or at 'elbow' point
        print("Transforming epoched data to principal component (PC) space")
        hmp_data = hmp.utils.transform_data(self.epoch_data)

        # Keep conditions empty to train HMP model on all data, add conditions to separate them
        # this is useful when conditions cause different stages or stage lengths
        if len(self.conditions) > 0:
            model_labels = None
            for condition in self.conditions:
                condition_subset = hmp.utils.condition_selection(
                    hmp_data,
                    self.epoch_data,
                    condition,
                    variable=self.condition_variable,
                )

                # Determine amount of expected events from number of supplied labels
                if self.fit_function == "fit_single":
                    self.fit_args["n_events"] = len(self.labels[condition])

                # Fit
                print(f"Fitting HMP model for {condition} condition")
                model = self.__fit_model__(condition_subset)

                # Label
                print(f"Labeling dataset for {condition} condition")
                new_labels = self.__label_model__(model, condition)
                if model_labels is None:
                    model_labels = new_labels
                else:
                    # Merge new labels with old labels, will always be disjoint sets since an epoch can only be one condition
                    model_labels = np.where(
                        model_labels == "", new_labels, model_labels
                    )
        else:
            print("Fitting HMP model")
            model = self.__fit_model__(hmp_data)
            print("Labeling dataset")
            model_labels = self.__label_model__(model)

        # Add label information to stage_data: ['', '', 'stage1', 'stage1', 'stage2' ...]
        stage_data = self.epoch_data.assign(
            labels=(["participant", "epochs", "samples"], model_labels)
        )
        return stage_data

    def __fit_model__(self, hmp_data):
        # Initialize model
        model = hmp.models.hmp(
            hmp_data, self.epoch_data, cpus=self.cpus, sfreq=self.epoch_data.sfreq
        )

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
        labels = self.labels if condition is None else self.labels[condition]

        if len(labels) != n_events:
            raise ValueError(
                "Amount of labels is not equal to amount of events, adjust labels parameter"
            )

        # Set up output datatypes
        event_locations = model.eventprobs.idxmax(dim="samples").astype(int)

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
            epoch = int(condition_epochs[data[1]])
            participant = participants.index(data[0])
            if participant != prev_participant:
                print(f"Processing participant {data[0]}")

            # TODO Maybe not reliable enough, what if electrode 0 (Fp1) is working but others are not
            # Find first sample from the end for combination of participant + epoch where the value is NaN
            # this is the reaction time sample where the participant pressed the button and stage ends
            RT_data = self.epoch_data.sel(
                participant=data[0], epochs=epoch, channels="Fp1"
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
                # Slice from known event location n to known event location n + 1
                # unless it is the last event, then slice from known event location n to reaction time
                if j != n_events - 1:
                    if not locations[j + 1] > RT_sample - 1:
                        samples_slice = slice(location, locations[j + 1])
                    else:
                        # End of stage is later than NaNs begin
                        continue
                else:
                    if not location > RT_sample - 1:
                        samples_slice = slice(location, RT_sample - 1)
                    else:
                        # NaNs begin before beginning of stage, error in measurement, disregard stage
                        continue
                if self.verbose:
                    print(
                        f"Participant: {participant}, Epoch: {epoch}, Sample range: {samples_slice}, Reaction time sample: {RT_sample}"
                    )
                labels_array[participant, epoch, samples_slice] = labels[j]

        return np.copy(labels_array)
