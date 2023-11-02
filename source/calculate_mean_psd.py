import pandas as pd
import numpy as np
from scipy.signal import welch

def calculate_mean_psd(data_df, fs, nperseg, noverlap, frequency_ranges):
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
    mean_psd_results = {}

# Iterate through the columns (channels) of the DataFrame
    for channel in data_df.columns[:len(data_df.columns)-5]:
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

