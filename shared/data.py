# Select data from path, can select based on conditions or participants, by default all participants are selected
class DataWrangler():
    # List of data sets that should be loaded and kept in memory
    def __init__(paths):
        ...

    def select_data(path, conditions, participants=None):

        ...

    # Input iterable of datasets (output of select_data), return combined version
    def combine_data(sets):
        ...