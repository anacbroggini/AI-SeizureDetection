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
import joblib
import tempfile
import os



st.title("Sa.i.zure")
st.markdown("AI based seizure detection")
st.write('by Ana, Silvan, Tassilo and Samet')
st.header("Data Exploration")
st.markdown("EEG data from CHb-MIT dataset --add more here--")

from PIL import Image

# Checkboxes to toggle visibility
show_visualization1 = st.sidebar.checkbox("Show Channels Frequency", value=True)
show_visualization2 = st.sidebar.checkbox("Show Variance plot", value=True)

st.subheader("A Visualization of the channels contained in the EEG Dataset")

# Content to be toggled
if show_visualization1:
    
    
    image1 = Image.open('/Users/Samet/Documents/Bildung/BootcampSpiced/github_rice_regression/Capstone_project/Channels Frequency.png')
    st.image(image1, caption='Overlay of Channels on Amplitude/Time axis', use_column_width=True)

st.subheader('Variance plot top ten True/False Seizures')
if show_visualization2:
    image2 = Image.open('/Users/Samet/Documents/Bildung/BootcampSpiced/github_rice_regression/Capstone_project/Variance plot top ten TrueFalse Seizures.png')
    st.image(image2, caption='Variance plot top ten True/False Seizures', use_column_width=True)

st.text("")
st.text("Dataset description --add more here--")

"""
# My first Classification ML app
Here's our first attempt at using data to create a classification report from User Inpuy:
"""

"""## Input parameters"""



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

            # Read the EDF file and convert to DataFrame (you can use the preprocessing function here)
            edf_df_classifier = read_edf_file(temp_filepath)

            # Perform classification using the loaded model
            # ...

            # Display the classification result
            st.subheader('Classification Result:')
            # ...

        except Exception as e:
            st.error(f'An error occurred during classification: {e}')
        finally:
            # Remove the temporary file
            os.remove(temp_filepath)

if __name__ == '__main__':
    main()



