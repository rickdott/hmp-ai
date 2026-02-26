import os
from pathlib import Path

DATA_PATH = Path(os.getenv("DATA_PATH"))

# Path can also be a list (if participants are split up across multiple datasets from same experiment)
DESC_STOP1 = {
    "path": DATA_PATH / "stopsignal" / "labelled_data_200hz.nc",
    "label": ["go1", "go2", "go3", "go4"],
    "strategy": "stop1",
    "info_to_keep": [], 
    "subset_cond": ('grouped_rt', 'equal', ['go/success/low', 'go/success/high'])
}

DESC_PRP1 = {
    "path": DATA_PATH / "prp" / "labelled_data_200hz_t1.nc",
    "label": ["t1_1", "t1_2", "t1_3", "t1_4"],
    "strategy": "prp/t1",
    "info_to_keep": [],
    "share_participants_with": "prp/t2",
    "subset_cond": ("condition", "equal", "long")
}

DESC_PRP2 = {
    "path": DATA_PATH / "prp" / "labelled_data_200hz_t2.nc",
    "label": ["t2_1", "t2_2", "t2_3"],
    "strategy": "prp/t2",
    "info_to_keep": [],
    "share_participants_with": "prp/t1",
    "subset_cond": ("condition", "equal", "long")
}

DESC_SAT1 = {
    "path": DATA_PATH / "sat1" / "preprocessed_500hz" / "labelled_data_200hz.nc",
    "label": ["encoding", "decision", "confirmation", "response"],
    "strategy": ["sat1/speed", "sat1/accuracy"],
    "info_to_keep": [],
    # "subset_cond": ("condition", "equal", "long") # No need to subset force == high since we do that when estimating
}
# DESC_CONF1 = {
#     "path": DATA_PATH / "conf1" / "labelled_data_200hz.nc",
#     "label": ["s1_op1", "s1_op2", "s1_op3", "s1_op4", "s2_op1", "s2_op2", "s2_op3", "s2_op4", "s2_op5"],
#     "strategy": ["rdk/s1", "rdk/s2"],
#     "info_to_keep": ["confidence"],
# }
DESC_CONF1 = {
    "path": DATA_PATH / "conf1" / "labelled_data_200hz.nc",
    "label": ["conf1_op1", "conf1_op2", "conf1_op3", "conf1_op4"],
    "strategy": "conf1",
    "info_to_keep": ["confidence"],
}

DESC_CONF2 = {
    "path": DATA_PATH / "conf2" / "labelled_data_200hz.nc",
    "label": ["conf2_op1", "conf2_op2", "conf2_op3", "conf2_op4", "conf2_op5"],
    "strategy": "conf2",
    "info_to_keep": ["confidenceRating"],
}

DESC_CONF3_EXP1 = {
    "path": DATA_PATH / "conf3_exp1" / "labelled_data_200hz.nc",
    "label": ["conf3_exp1_op1", "conf3_exp1_op2", "conf3_exp1_op3", "conf3_exp1_op4"],
    "strategy": "conf3_exp1",
    "info_to_keep": ["confidence"],
}

DESC_CONF3_EXP2 = {
    "path": DATA_PATH / "conf3_exp2" / "labelled_data_200hz.nc",
    "label": ["conf3_exp2_op1", "conf3_exp2_op2", "conf3_exp2_op3", "conf3_exp2_op4"],
    "strategy": "conf3_exp2",
    "info_to_keep": ["conf"],
}

DESC_CONF3_EXP3 = {
    "path": DATA_PATH / "conf3_exp3" / "labelled_data_200hz.nc",
    "label": ["conf3_exp3_op1", "conf3_exp3_op2", "conf3_exp3_op3", "conf3_exp3_op4", "conf3_exp3_op5"],
    "strategy": "conf3_exp3",
    "info_to_keep": ["RTchoice", "confidence"],
    "rt_key": "RTchoice",
}

