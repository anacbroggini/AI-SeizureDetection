import pandas as pd
import numpy as np
from scipy.signal import welch
from source.constants import FREQUENCY_RANGES
from source.constants import CHANNELS

def calculate_mean_psd(data_df, fs=256, nperseg=256, noverlap=128, frequency_ranges=FREQUENCY_RANGES):
    """
    Calculate mean Power Spectral Density (PSD) for each channel in the DataFrame
    within specified frequency ranges.

    Parameters:
        data_df (pd.DataFrame): DataFrame containing EEG data with channels as columns.
        fs (int): Sampling frequency.
        nperseg (int): Length of each segment for PSD calculation.
        noverlap (int): Overlap between segments for PSD calculation.
        frequency_ranges (dict): Dictionary of frequency range limits (e.g., {'Delta': (0.5, 4), ...}).

    Returns:
        dict: A dictionary of mean PSD results for each channel and frequency range.
    """
    
    if type(data_df) is pd.core.frame.Series:
        data_df = pd.DataFrame(data_df)

    # if fs is None:
    #     if data_df.index.freq is None:
    #         freq = pd.infer_freq(data_df.index)
    #         data_df.index.freq = pd.tseries.frequencies.to_offset(freq)
    #     fs = 1.0  (data_df.index.freq / 10E9)
    #     print(fs)

    mean_psd_results = {}


    # Iterate through the columns (channels) of the DataFrame
    for channel in CHANNELS:
        if channel not in data_df.columns:
            continue
        # Calculate the PSD for the current channel
        f, Pxx = welch(data_df[channel], fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density")

        # Initialize an empty dictionary to store mean PSD values for each frequency range
        channel_mean_psd = {}

        # Calculate mean PSD in each frequency range
        for band_name, (low_freq, high_freq) in frequency_ranges.items():
            # Find the indices corresponding to the specified frequency range
            indices = np.where((f >= low_freq) & (f <= high_freq))

            # Calculate the mean PSD within the range
            mean_psd = np.mean(Pxx[indices])

            # Store the mean PSD value for the current frequency range
            channel_mean_psd[band_name] = mean_psd

        # Store the mean PSD results for the current channel in the dictionary
        mean_psd_results[channel] = channel_mean_psd
        
    return mean_psd_results

