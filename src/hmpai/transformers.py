"""Transforms epoched data using a custom linear combination for HMP analysis.

Projects channels into a custom space given by a n_chan x n_component matrix.
 See example matrix when applying `ProjPCA` class.
"""
from typing import Optional

import xarray as xr

from hmp.transformers.base import BaseTransformer


class ProjCustomKeepData(BaseTransformer):
    """Transforms epoched data using a a custom linear combination for HMP analysis.

    Projects channels into a custom space given by a n_chan x n_component matrix.
     See example matrix when applying `ProjPCA` class.

    Parameters
    ----------
    epoch_data : xr.Dataset
        Input EEG data with dimensions [participant, epoch, sample, channel], from `io` module
    weights: xr.DataArray
        Custom linear combination of channels as an xarray.DataArray with 'channel' and 'component'
         dimensions and the weights
    interval_id: str
        Name of the variable that contains the trial intervals in the epoch_data used for cropping.
    offset_after_end : float
        Time offset after interval start for cropping.
    min_duration : float, optional
        Minimum duration threshold for keeping epochs.
    max_duration : float, optional
        Maximum duration threshold for keeping epochs.
    reject_threshold : float, optional
        Threshold for rejecting noisy epochs.
    center : bool
        Whether to center the data across the last dimension before projection
    whiten : bool
        Return the components with unit-variance
    common_variance : bool
        Whether to standardize variance across trials.
    subject_zscore: bool
        Z-score each component for each participant
    subject_zscore: bool
        Participant-wise standardization of the projection components using zscores
    verbose : bool
        Whether to print rejection/cropping details.
    """

    def __init__(#noqa: PLR0913
        self,
        epoch_data: xr.Dataset,
        weights: xr.DataArray,
        interval_id: str = 'rt',
        offset_after_end: float = 0,
        offset_before_start: float = 0,
        min_duration: float = 0,
        max_duration: float = float('Inf'),
        reject_threshold: Optional[float] = None,
        center: bool = True,
        whiten: bool = True,
        common_variance: bool = False,
        subject_zscore: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            interval_id=interval_id,
            offset_after_end=offset_after_end,
            offset_before_start=offset_before_start,
            min_duration=min_duration,
            max_duration=max_duration,
            reject_threshold=reject_threshold,
            verbose=verbose,
            common_variance=common_variance,
            subject_zscore=subject_zscore,
            whiten=whiten,
            center=center,
        )
        # Preprocessing
        attrs = epoch_data.attrs
        data = self.common_preprocess(epoch_data)
        
        self.data_epoched = data.copy()
        self.data_epoched = self.data_epoched.unstack().to_dataset(name="data").transpose("participant", "epoch", "channel", "sample")
        self.data_epoched.attrs = attrs

        # Final formatting
        self.data_format(
            data, weights
        )

