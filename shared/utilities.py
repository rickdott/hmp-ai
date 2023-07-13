import tensorflow as tf
import numpy as np

SAT1_STAGES_ACCURACY = ['encoding', 'decision', 'confirmation', 'response']
SAT1_STAGES_SPEED = ['encoding', 'decision', 'response']
SAT2_STAGES_ACCURACY = ['encoding', 'decision', 'confirmation', 'response']
SAT2_STAGES_SPEED = ['encoding', 'decision', 'response']
AR_STAGES = ['encoding', 'familiarity', 'memory', 'decision', 'response']

# Takes a model and the probabilities of events occuring within the dataset the model was initiated on
# and returns an ndarray of shape samples x time x #electrodes
# length of labels must be equal to amount of events
def label_model(model, eeg_data, labels):
    n_events = len(model.event)
    if len(labels) != n_events:
        raise ValueError('Amount of labels is not equal to amount of events, adjust labels parameter')
    
    # Set up output datatypes
    event_locations = model.eventprobs.idxmax(dim='samples').astype(int)
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
            print(f'Processing participant {data[0]}')
        
        # TODO Maybe not reliable enough, what if electrode 0 (Fp1) is working but others are not
        # Find sample for combination of participant + epoch where the value is null, this is the reaction time sample
        # where the participant pressed the button and the last stage ends
        RT_sample = int(eeg_data.sel(participant=data[0], epochs=data[1]).isnull().argmax('samples').data[0])
        prev_participant = participant
        epoch = data[1]

       # Set stage label for each stage
        for j, location in enumerate(locations):
            # Slice from known event location n to known event location n + 1
            # unless it is the last event, then slice from known event location n to reaction time
            samples_slice = slice(location, locations[j + 1]) if j != n_events - 1 else slice(location, RT_sample - 1)
            labels_array[participant, epoch, samples_slice] = labels[j]

    return labels_array

earlyStopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)