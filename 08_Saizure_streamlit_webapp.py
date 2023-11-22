import streamlit as st
import pandas as pd
import numpy as np
import mne
import pickle 
from pickle import dump
from mne.io import  read_raw_edf
import matplotlib.pyplot as plt
import seaborn as sns
from source import data_import
from source.filter_eeg_channels_web import filter_eeg_channels
import joblib
import tempfile
import os
from source.constants import CHANNELS, FREQUENCY_RANGES
from source.extract_features import extract_features
import scipy
from scipy.signal import butter, lfilter

st.title("Sa.i.zure")
st.markdown("AI based seizure detection")
st.write('by Ana, Silvan, Tassilo and Samet')
#st.header("Data Exploration")
#st.markdown("EEG data from CHb-MIT dataset --add more here--")

from PIL import Image

# Load the saved classification model
model_path = "Class_models/best_xgboost_model.pkl"  
loaded_model = joblib.load(model_path)

st.text("EDF File requirements:")
channels = "'F4-C4', 'F3-C3', 'FT9-FT10', 'FZ-CZ', 'F7-T7', 'FP2-F4', 'T8-P8-1', 'T8-P8-0', 'FP1-F3', 'CZ-PZ'"
st.text(f"The edf file should contain channels: {channels} ")
st.text("and a sampling rate of 256 Hz")

"""
# Classification  Webapp 
This our working prototype of a classification webapp from User uploaded edf files:
"""

# Function to read EDF file and convert to DataFrame
def read_edf_file(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    df = raw.to_data_frame()
    return df



# Define the Streamlit app
def main():
    st.title('EDF File Viewer and Classifier')

    # User file upload for EDF files (Viewer)
    uploaded_file_viewer = st.file_uploader('Upload an EDF file for viewing:', type=['edf'], key='viewer')

    """## This is the fun part, we are going to detect seizures in the data. """
    """Compared to the traditional method of identifying seizures manually by hand from an expert, """
    """We are going to use a machine learning model to do this for us in a much faster time. :)"""
  
    # User file upload for EDF files (Classifier)
    uploaded_file_classifier = st.file_uploader('Upload an EDF file for classification:', type=['edf'], key='classifier')

    # Process user input (Viewer)
    if uploaded_file_viewer is not None:
        try:
            # Save the uploaded EDF file to a temporary location
            with st.spinner('Reading and processing file for viewing...'):
                temp_filepath = os.path.join(tempfile.gettempdir(), "temp_file.edf")
                temp_file = open(temp_filepath, 'wb')
                temp_file.write(uploaded_file_viewer.read())
                temp_file.close()

            # Read the EDF file and convert to DataFrame
            edf_df_viewer = read_edf_file(temp_filepath)

            # Display the DataFrame (Viewer)
            st.subheader('EDF File Contents for Viewing:')
            st.dataframe(edf_df_viewer)

        except Exception as e:
            st.error(f'An error occurred during viewing: {e}')
        finally:
            # Remove the temporary file
            os.remove(temp_filepath)

   

    # Process user input (Classifier)
    if uploaded_file_classifier is not None:
        try:
            # Save the uploaded EDF file to a temporary location
            with st.spinner('Reading and processing file for classification...'):
                temp_filepath = os.path.join(tempfile.gettempdir(), "temp_file.edf")
                temp_file = open(temp_filepath, 'wb')
                temp_file.write(uploaded_file_classifier.read())
                temp_file.close()

            # Read and preprocess the EDF file
            edf_df_classifier = data_import.load_segmented_unlabeled_data(temp_filepath, channels=CHANNELS)
            

            exclude_ranges=[[58, 62], [118, 122]]
            filtered = filter_eeg_channels(edf_df_classifier, CHANNELS, fs=256, exclude_ranges=exclude_ranges, Q=30)
                
            # Extract features from the preprocessed data
            extracted_features = extract_features(filtered)

            # Perform classification using the loaded model
            predictions = loaded_model.predict(extracted_features)
            time = list(range(0, len(predictions)))
            signal = extracted_features["F7-T7_mean"].values

            tmp = {"predictions": predictions,
                   "time": time,
                   "signal": signal}

            
            plot_df = pd.DataFrame(tmp)

            # Initializing variables
            seizure_number = 0
            start_seizure = None
            end_seizure = None

            seizure_intervals = []

            # Iterating through the DataFrame to identify seizure intervals
            for index, row in plot_df.iterrows():
                if row['predictions']:
                    if start_seizure is None:
                        start_seizure = row['time']
                        seizure_number += 1
                else:
                    if start_seizure is not None:
                        end_seizure = row['time']
                        seizure_intervals.append((seizure_number, start_seizure, end_seizure))
                        start_seizure = None
                        end_seizure = None

            # Creating a DataFrame with seizure intervals
            seizure_df = pd.DataFrame(seizure_intervals, columns=['seizure_number', 'start_seizure', 'end_seizure'])

            # Custom color values
            amplitude_color = '#5a4275'
            background_color = "white"
            grid_color = '#977cca'
            axvspan_color = 'red'

            # Plot the selected sensor's EEG data
            fig = plt.figure(figsize=(10, 6))
            plt.plot(plot_df["time"], plot_df["signal"], color=amplitude_color)  # Change amplitude color
            plt.title(f'Mean Amplitude of F7-T7 over Time', color=grid_color, fontweight='bold')
            plt.xlabel('Seizure Sequence (Time x 5 Seconds)', color=grid_color, fontweight='bold')
            plt.ylabel('Amplitude', color=grid_color, fontweight='bold')
            plt.grid(True, color=grid_color)  # Set grid color
            plt.gca().set_facecolor(background_color)  # Set background color to white

            # Iterate through the seizure df to plot a shade of red on the amplitude for each seizure sequence
            for index, row in seizure_df.iterrows():
                seizure_start = row['start_seizure']
                seizure_end = row['end_seizure']
                plt.axvspan(seizure_start, seizure_end, color=axvspan_color, alpha=0.3, label=f'Seizure {row["seizure_number"]}')

            # Change font properties for tick labels and set their color
            plt.xticks(fontweight='bold', color=grid_color)
            plt.yticks(fontweight='bold', color=grid_color)

            # Remove frame around the plot
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.tick_params(axis='both', which='both', length=0)

            plt.tight_layout()
            #plt.savefig('amplitude_transparent_bg.png', transparent=True)
            plt.show()

            st.pyplot(fig)
            # Here we create a new df which counts the number of seizures and when they started and ended (in seconds)
            
            # Filter out rows with a difference of 6 between start_time and end_time
            filtered_df = seizure_df[seizure_df['end_seizure'] - seizure_df['start_seizure'] >= 6]
            
            # Apply post-processing to identify seizures
            seizure_threshold = 6
            seizure_detected = np.convolve(predictions, np.ones(seizure_threshold), mode='valid') >= seizure_threshold
            
            # Find all occurrences of seizures
            seizure_occurrences = np.where(seizure_detected)[0]
            
            # Display the classification result
            st.subheader('Classification Result:')

            # Display the classification result        
            if any(seizure_detected):
                st.header("Seizure detected !!!")

                # Display the filtered_df DataFrame in Streamlit
                # st.subheader('Seizure Intervals:')
                # st.dataframe(filtered_df)

                # Write the seizure start and end times
                for index, row in filtered_df.iterrows():
                    start_time = row['start_seizure']
                    end_time = row['end_seizure']
                    st.write(f"Seizure starts at {start_time * 5} seconds and ends at {end_time * 5} seconds")

                # detection_index = np.argmax(seizure_detected)
                # detection_time_stamp = detection_index * 5
                # st.write(f'Detection Timestamp: {detection_time_stamp} seconds')
            else:
               st.header("No seizure")

        except Exception as e:
            st.error(f'An error occurred during classification: {e}')
        finally:
            # Remove the temporary file
            os.remove(temp_filepath)

if __name__ == '__main__':
    main()



