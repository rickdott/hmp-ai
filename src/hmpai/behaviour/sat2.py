from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

SAT2_SPLITS = [
    ["S6", "S2", "S16", "S18", "S12", "S9", "S13", "S1", "S20"],  # Train
    ["S5", "S11", "S15", "S17", "S14"],  # Test
    ["S4", "S10", "S3", "S7", "S8"],  # Val
]


def read_behavioural_info(path: Path):
    """
    Reads behavioral information from a CSV file, removes specific indexing columns, 
    and returns the processed data.

    Args:
        path (Path): The file path to the CSV file containing behavioral data.

    Returns:
        pandas.DataFrame: A DataFrame containing the behavioral data with indexing 
        columns removed. Specifically, the first column and the ninth column 
        (index 0 and 8) are dropped.
    """
    data = pd.read_csv(path)

    # Remove indexing columns
    data = data.drop([data.columns[0], data.columns[8]], axis=1)
    return data


def match_on_event_name(
    event_name: str,
    behaviour_groups: pd.Series | pd.DataFrame,
    participant: str,
    rt: float,
):
    """
    Matches an event based on its name and retrieves the closest reaction time (RT) 
    from the provided behaviour groups.

    Args:
        event_name (str): The event name string, formatted as "prefix/force/sat/expdResp/contrast".
        behaviour_groups (pd.Series | pd.DataFrame): A pandas Series or DataFrame containing 
            behavioural data. If a Series, it is expected to be a grouped object.
        participant (str): The participant identifier.
        rt (float): The reaction time (RT) in seconds.

    Returns:
        pd.Series: A pandas Series representing the row in the behaviour groups that 
        matches the event and has the closest RT. Returns an empty Series if no match is found.

    Raises:
        KeyError: If the event_name format is invalid or required columns are missing 
        in the behaviour_groups DataFrame.
    """
    # Extract relevant parts of the event_name
    aspects = event_name.split("/")
    force = aspects[1]  # Not used
    sat = aspects[2]
    expdResp = aspects[3]
    contrast = int(aspects[4])

    # Access the relevant group
    if type(behaviour_groups) != pd.DataFrame:
        try:
            rows = behaviour_groups.get_group((participant, sat, expdResp, contrast))
        except KeyError:
            return pd.Series()  # Handle cases where no match is found
    else:
        rows = behaviour_groups[
            (behaviour_groups["participant"] == participant)
            & (behaviour_groups["SAT"] == sat)
            & (behaviour_groups["expdResp"] == expdResp)
            & (behaviour_groups["contrast"] == int(contrast))
        ]

    # Find the closest RT
    closest_idx = (rows["rt"] - (rt * 1000)).abs().idxmin()
    return rows.loc[closest_idx, :]


def merge_data(true_df: pd.DataFrame, behaviour: pd.DataFrame):
    """
    Merges two DataFrames by matching rows from `true_df` with processed rows from `behaviour`.

    Args:
        true_df (pd.DataFrame): DataFrame containing ground truth data with columns including 
                                'event_name', 'participant', and 'rt'.
        behaviour (pd.DataFrame): DataFrame containing behavioral data with columns including 
                                  'participant', 'SAT', 'expdResp', and 'contrast'.

    Returns:
        pd.DataFrame: A merged DataFrame with matched rows from `true_df` and processed `behaviour`, 
                      dropping duplicate participant columns and renaming appropriately.
    """
    def preprocess_behaviour(behaviour: pd.DataFrame):
        return behaviour.groupby(["participant", "SAT", "expdResp", "contrast"])

    def get_matching_row(row, behaviour):
        return match_on_event_name(
            row["event_name"], behaviour, row["participant"], row["rt"]
        )

    behaviour = preprocess_behaviour(behaviour)

    matched_df = true_df.apply(lambda row: get_matching_row(row, behaviour), axis=1)
    merged = true_df.reset_index(drop=True).merge(
        matched_df.reset_index(drop=True), left_index=True, right_index=True
    )
    merged = merged.drop(columns=["participant_y"])
    merged = merged.rename(columns={"participant_x": "participant"})
    return merged


def merge_data_xr(epoch_data: xr.Dataset, behaviour: pd.DataFrame):
    """
    Merges behavioural data into an xarray Dataset by aligning on event names and reaction times.

    Parameters:
        epoch_data (xr.Dataset): An xarray Dataset containing epoch data with coordinates 
                                 'event_name', 'rt', and 'participant'.
        behaviour (pd.DataFrame): A pandas DataFrame containing behavioural data, grouped by 
                                  'participant', 'SAT', 'expdResp', and 'contrast'.

    Returns:
        xr.Dataset: The updated xarray Dataset with new coordinates ('givenResp', 'trialType') 
                    added based on the behavioural data.
    """

    columns_to_add = ["givenResp", "trialType"]

    behaviour = behaviour.groupby(["participant", "SAT", "expdResp", "contrast"])

    # (participant, epochs) ndarray
    event_names = epoch_data.coords["event_name"].values
    rts = epoch_data.coords["rt"].values

    new_coords = {col: np.full_like(event_names, np.nan) for col in columns_to_add}

    for participant in range(event_names.shape[0]):
        for epoch in range(event_names.shape[1]):
            event_name = event_names[participant, epoch]
            rt = rts[participant, epoch]
            if np.isnan(rt):
                continue

            participant_name = epoch_data.coords["participant"].values[participant]
            behaviour_row = match_on_event_name(
                event_name, behaviour, participant_name, rt
            )
            for column in columns_to_add:
                if column in behaviour_row:
                    new_coords[column][participant, epoch] = behaviour_row[column]

    for col, values in new_coords.items():
        epoch_data = epoch_data.assign_coords(
            {col: (("participant", "epochs"), values)}
        )
    return epoch_data
