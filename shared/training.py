from shared.generators import NewSAT1DataGenerator
import random
import datetime
from shared.utilities import get_summary_str, earlyStopping_cb, LoggingTensorBoard
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


def split_data_on_participants(data, train_percentage=60):
    random.seed(42)
    participants = data.participant.values.tolist()
    # In case of SAT1 experiment, 25 participants are used
    # Given train_percentage=60, remaining 40 percent will be split evenly between validation and test sets
    # 100-train_percentage must be divisible by 2

    # Find amounts of train and test/val participants
    train_n = int(len(participants) * (train_percentage / 100))
    testval_n = len(participants) - train_n

    # Split into train, test, and val by sampling randomly
    testval_participants = random.sample(participants, testval_n)
    train_participants = [p for p in participants if p not in testval_participants]
    val_participants = testval_participants[: testval_n // 2]
    test_participants = testval_participants[testval_n // 2 :]

    # Select subsets from data
    train_data = data.sel(participant=train_participants)
    val_data = data.sel(participant=val_participants)
    test_data = data.sel(participant=test_participants)

    return train_data, val_data, test_data


def train_and_evaluate(
    model, train, val, test, batch_size=16, epochs=20, workers=8, additional_info=dict()
):
    # Create generators
    train_gen = NewSAT1DataGenerator(train, batch_size)
    val_gen = NewSAT1DataGenerator(val, batch_size)
    test_gen = NewSAT1DataGenerator(test, batch_size)

    # Set up configuration logging
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path("logs/") / run_id
    to_write = {"Model summary": get_summary_str(model)}
    to_write.update(additional_info)

    fit = model.fit(
        train_gen,
        epochs=epochs,
        callbacks=[earlyStopping_cb, LoggingTensorBoard(to_write, log_dir=path)],
        validation_data=val_gen,
        use_multiprocessing=True,
        workers=workers,
    )

    # Test model and write test summary
    writer = tf.summary.create_file_writer(str(path / "train"))
    predicted_classes = np.argmax(model.predict(test_gen), axis=1)
    predicted_classes = [test_gen.cat_labels[idx] for idx in list(predicted_classes)]
    test_results = classification_report(test_gen.full_labels, predicted_classes).splitlines()
    test_results = "    " + "\n    ".join(test_results)

    with writer.as_default():
        tf.summary.text('Test results', test_results, step=0)

    return fit
