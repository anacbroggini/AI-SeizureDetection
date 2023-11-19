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
st.header("Data Exploration")
st.markdown("EEG data from CHb-MIT dataset --add more here--")

from PIL import Image

# Load the saved classification model
model_path = "Class_models/best_xgboost_model.pkl"  
loaded_model = joblib.load(model_path)

# Checkboxes to toggle visibility
show_visualization1 = st.sidebar.checkbox("Show Channels Frequency", value=True)
show_visualization2 = st.sidebar.checkbox("Show Variance plot", value=True)

st.subheader("A Visualization of the channels contained in the EEG Dataset")

# Content to be toggled
if show_visualization1:
    
    
<<<<<<< HEAD
    image1 = Image.open('./source/Channels_Frequency.png')
=======
    image1 = Image.open('Images/Channels_Frequency.png')
>>>>>>> c7973de (cleanup)
    st.image(image1, caption='Overlay of Channels on Amplitude/Time axis', use_column_width=True)

st.subheader('Variance plot top ten True/False Seizures')
if show_visualization2:
<<<<<<< HEAD
    image2 = Image.open('./source/Variance_plot_top_ten.png')
=======
    image2 = Image.open('Images/Variance_plot_top_ten.png')
>>>>>>> c7973de (cleanup)
    st.image(image2, caption='Variance plot top ten True/False Seizures', use_column_width=True)

st.text("")
st.text("Dataset description --add more here--")

"""
# Classification  Webapp 
Here's our first attempt at using data to create a classification report from User Inpuy:
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

    """## This is the fun part, we are going to use a classifier on the data and predict the outcome. """
    """Precisely we want to predict if the patient has a seizure or not. :) """
    
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

            # Apply post-processing to identify seizures
            seizure_threshold = 6
            seizure_detected = np.convolve(predictions, np.ones(seizure_threshold), mode='valid') >= seizure_threshold

            # Display the classification result
            st.subheader('Classification Result:')
            col1, col2 = st.columns(2)
            col1.write(predictions)
            col2.write(seizure_detected)

            # Print "Seizure detected!!!" if a seizure is detected
            
            if any(seizure_detected):
                st.subheader('Result:')
                st.header("Seizure detected !!!")


        except Exception as e:
            st.error(f'An error occurred during classification: {e}')
        finally:
            # Remove the temporary file
            os.remove(temp_filepath)

if __name__ == '__main__':
    main()



