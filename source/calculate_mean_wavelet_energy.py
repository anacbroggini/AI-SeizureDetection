import pandas as pd
import numpy as np
from scipy.signal import cwt, ricker
from source.constants import FREQUENCY_RANGES
from source.constants import CHANNELS

def calculate_mean_wavelet_energy(data_df, fs=256, frequency_ranges=FREQUENCY_RANGES):
    """
    Calculate mean wavelet energy for each channel in the DataFrame
    within specified frequency ranges.

    Parameters:
        data_df (pd.DataFrame): DataFrame containing EEG data with channels as columns.
        fs (int): Sampling frequency.
        frequency_ranges (dict): Dictionary of frequency range limits (e.g., {'Delta': (0.5, 4), ...}).

    Returns:
        dict: A dictionary of mean wavelet energy results for each channel and frequency range.
    """
    
    if type(data_df) is pd.core.frame.Series:
        data_df = pd.DataFrame(data_df)

    mean_wavelet_energy_results = {}

    # Iterate through the columns (channels) of the DataFrame
    for channel in CHANNELS:
        if channel not in data_df.columns:
            continue

        # Calculate the continuous wavelet transform for the current channel
        widths = np.arange(1, fs)
        cwt_result = cwt(data_df[channel], ricker, widths)

        # Initialize an empty dictionary to store mean wavelet energy values for each frequency range
        channel_mean_wavelet_energy = {}

        # Calculate mean wavelet energy in each frequency range
        for band_name, (low_freq, high_freq) in frequency_ranges.items():
            # Find the indices corresponding to the specified frequency range
            indices = np.where((widths >= low_freq) & (widths <= high_freq))

            # Calculate the mean wavelet energy within the range
            mean_wavelet_energy = np.mean(np.abs(cwt_result[indices])**2, axis=1)

            # Store the mean wavelet energy value for the current frequency range
            channel_mean_wavelet_energy[band_name] = np.mean(mean_wavelet_energy)

        # Store the mean wavelet energy results for the current channel in the dictionary
        mean_wavelet_energy_results[channel] = channel_mean_wavelet_energy
        
    return mean_wavelet_energy_results