import pandas as pd
from scipy.signal import butter, lfilter

def filter_eeg_channels(df, channels, fs=256, exclude_ranges=None, Q=30):
    """
    Filter EEG channels in a DataFrame using band-stop filters.

    Parameters:
    - df: DataFrame containing EEG data.
    - channels: List of channel names to filter.
    - fs: Sampling frequency (default is 256 Hz).
    - exclude_ranges: List of exclusion ranges, e.g., [[58, 62], [118, 122]].
    - Q: Quality factor for notch filters (default is 30).

    Returns:
    - filtered_df: DataFrame with filtered EEG data.
    """

    filtered_df = df[channels].copy()
    filtered_df[['is_seizure','before_seizure']] = df[['is_seizure','before_seizure']]
    
    if exclude_ranges is None:
        exclude_ranges = []

    for channel in channels:
        for exclude_range in exclude_ranges:
            nyquist = 0.5 * fs
            low = (exclude_range[0] - 1.0) / nyquist
            high = (exclude_range[1] + 1.0) / nyquist

            b, a = butter(4, [low, high], btype='bandstop')
            filtered_eeg_data = lfilter(b, a, filtered_df[channel])
            filtered_df[channel] = filtered_eeg_data

    return filtered_df