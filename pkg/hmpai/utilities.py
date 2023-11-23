import numpy as np
import json


# Channel configuration for topological layout, "NA" means 'not available' and should not be used in training
CHANNELS_2D = np.array(
    [
        ["NA", "Fp1", "NA", "Fp2", "NA"],
        ["NA", "NA", "AFz", "NA", "NA"],
        ["F7", "F3", "Fz", "F4", "F8"],
        ["FC5", "FC1", "FCz", "FC2", "FC6"],
        ["T7", "C3", "Cz", "C4", "T8"],
        ["CP5", "CP1", "CPz", "CP2", "CP6"],
        ["P7", "P3", "Pz", "P4", "P8"],
        ["NA", "O1", "NA", "O2", "NA"],
    ],
    dtype=str,
)

# Channel order of AR experiment electrodes, re-ordered to be as similar as possible to SAT1
REINDEX_CHANNELS_AR = [
    "FP1",
    "FP2",
    "FPZ",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
    "FC3",
    "FCZ",
    "FC4",
    "FT7",
    "FT8",
    "TP7",
    "CP3",
    "CPZ",
    "CP4",
    "TP8",
    "trash1",
    "trash2",
]

AR_SAT1_CHANNELS = [
    "T7",
    "CPZ",
    "O2",
    "FCZ",
    "O1",
    "P8",
    "P4",
    "T8",
    "F4",
    "C4",
    "FP1",
    "C3",
    "FP2",
    "P3",
    "F8",
    "P7",
    "F3",
    "F7",
    "PZ",
    "FZ",
    "CZ",
]

# Value that means data should not be used in training
MASKING_VALUE = 999


def pad_to_max_sample_length(array: np.array, max_sample_length: int) -> np.array:
    """Pads ndarray to given length, in this case
    the length of the largest sample.

    Args:
        array (np.array): Array to be padded.
        max_sample_length (int): Length of largest sample.

    Returns:
        np.array: Padded array
    """
    padding = ((0, 0), (0, max_sample_length - array.shape[1]))
    return np.pad(array, padding)


def pretty_json(data: dict) -> str:
    # From https://www.tensorflow.org/tensorboard/text_summaries
    json_data = json.dumps(data, indent=2)
    return "".join(f"\t{line}" for line in json_data.splitlines(True))


def print_results(results: dict) -> str:
    # From a list of test results to an aggregated accuracy and F1-Score
    accuracy = 0.0
    f1 = 0.0
    for result in results:
        accuracy += result["accuracy"]
        f1 += result["macro avg"]["f1-score"]

    print(f"Average Accuracy: {accuracy / len(results)}")
    print(f"Average F1-Score: {f1 / len(results)}")
