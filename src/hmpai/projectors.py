import numpy as np
import xarray as xr
from hmp.basedata import from_io, PCA
from hmp.projectors.custom import Custom

def defaultKeepData(
            epoch_data: xr.Dataset,
            duration_id: str = 'response_time',
            offsets: tuple | float = (0,0),
            center: bool = True,
            min_duration: float = 0,
            max_duration: float | None = None,
            reject_amplitude: float = np.inf,
            n_comp: float | None = None,
            whiten: bool = True,
            common_variance: bool = False,
            standardize_recording: bool = False,
            weights: xr.DataArray | None = None,
            verbose: bool = True,
            for_mamba: bool = False
    ):
    """
    Create a BaseData instance from data from io.

    Includes:
     - epoch cropping and rejection
     - PCA
     - variance operations.

    Parameters
    ----------
    epoch_data : xr.DataArray
        Data with dimensions [sample, component, trial], coordinates that
        describe the dataset including recording, subject, epoch, and a trial
        MultiIndex, and attributes sfreq and offset. Typically obtained
        through class method 'from_io(..)'.
    duration_id: str, optional
        Name of the variable that contains the trial intervals in the epoch_data
        used for cropping.
        Default = 'response_time'.
    offsets : tuple, float, optional
        Seconds of recording to keep before and after end of each epoch duration.
        First value refers to the times taken before epoch center and second value
        to the time kept after end. Should be positive. Used for padding the data
        before crosscorrelation. Adding template width / 2 is recommended.
        If float apply the offsets symmetrically.
        Default = 0
    center : bool
        Median center the data after cropping including baseline
        default = False
    min_duration : float, optional
        Minimum duration threshold for keeping epochs.
        Default = 0
    max_duration : float, optional
        Maximum duration threshold for keeping epochs.
        Default = Inf
    reject_amplitude : float, optional
        Amplitude threshold for rejecting noisy epochs.
        Default = None
    n_comp: int, optional
        Nr of components retained if > 1, otherwise (0 < n_comp < 1) nr of components
        explaining at least n_comp% variance are retained.  If None, user input requested.
        Default = None
    whiten : bool, optional
        Return the components with unit-variance
        Default = True
    common_variance : bool, optional
        Standardize variance across trials.
        Default = False
    standardize_recording: bool, optional
        Divide each component for each recording by its standard deviation
        Default = False
    verbose:
        Provide feedback on the different operations

    Returns
    -------
    BaseData
        An instance of BaseData using default preprocessing routine
    """
    base_data = from_io(epoch_data)

    base_data.crop_reject_epochs(
        duration_id=duration_id,
        offsets=offsets,
        center=center,
        min_duration=min_duration,
        max_duration=max_duration,
        reject_amplitude=reject_amplitude,
        verbose=verbose
    )

    # Keep epoched and cropped data for HMP-AI
    attrs = base_data.data.attrs
    base_data.data_epoched = base_data.data.unstack().to_dataset(name="data").transpose("recording", "epoch", "channel", "sample")
    base_data.data_epoched.attrs = attrs

    if for_mamba:
        return base_data
    if weights is not None:
        # Use custom weights for projection
        base_data.project(Custom(weights))
    else:
        base_data.project(PCA(n_comp=n_comp))
        base_data.apply_variance_ops(
            whiten=whiten,
            common_variance=common_variance,
            standardize_recording=standardize_recording,
        )

    return base_data