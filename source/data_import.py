#%%

from pathlib import Path
from click import pause
import numpy as np
import pandas as pd
import mne
import pwd

DATA_ROOT = Path(__file__).resolve().parent / "../data"


#%% get patient_list
def get_patient_list(root_dir=DATA_ROOT, patient_ids=None):
    '''get patient list from "chb"-directories in root_dir. optionally define patient_ids as list or get all if None.
    input:
        root_dir: directory to look for patient subdirs (chb..)
        patient_ids: list of integers that define the looked for patients.

    returns list of patientnames: []"chb01", "chb02" ...] of target directory
    '''
    
    root_dir = Path(root_dir)
    if patient_ids is not None:
        patient_list = [f'chb{id:02}' for id in patient_ids]
        return patient_list
    patient_list = sorted([d.name for d in root_dir.glob('chb*')])
    return patient_list

#%% get session list
def get_session_list(root_dir=DATA_ROOT, patient='chb01'):
    '''get filenames of sessions for specific patient, ie. ["chb01_01.edf", ...] '''

    root_dir = Path(root_dir)
    session_list = sorted([s.name for s in (root_dir / patient).rglob('*.edf')])
    return session_list

#%% get summary data

def get_patient_summary(root_dir=DATA_ROOT, patient='chb01'):
    '''read in the summary file of specified patient and return it as a list of dictionaries. One dictionary for each seizure.
    
    returns: list of dictionaries.
    '''

    file_path = Path(DATA_ROOT) / patient / (patient + "-summary.txt")
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("File Name:"):
                current_data = {
                    "File Name": None, 
                    "Number of Seizures": None, 
                    "Seizure Start Time": np.nan, 
                    "Seizure End Time": np.nan, 
                    "patient": patient
                    }
                current_data["File Name"] = line.split(": ", 1)[1]
            elif line.startswith("Number of Seizures in File:"):
                current_data["Number of Seizures"] = line.split(": ", 1)[1]
                if current_data["Number of Seizures"] == 0:
                    data.append(current_data.copy())
            elif line.startswith("Seizure Start Time:"):
                current_data["Seizure Start Time"] = int(line.split(": ", 1)[1].split()[0])
            elif line.startswith("Seizure End Time:"):
                current_data["Seizure End Time"] = int(line.split(": ", 1)[1].split()[0])
                data.append(current_data.copy())
    return data


### function import_eeg_data
#%%
def return_pandas_df(root_dir=DATA_ROOT, patient=None, session=None, target_freq=None, summary=None):
    ''' Read specific session .edf, transform to pandas dataframe with timedelta index, optional resampling, labeling seizures in "is_seizure" column.
    
    returns pandas dataframe of session.
    '''
    file_path = root_dir / patient / session

    raw = mne.io.read_raw_edf(file_path, preload=False, verbose="ERROR")
    
    sample_freq = str(1/raw.info['sfreq'] * 1E9) + 'N'
    df = raw.to_data_frame(index='time')
    time_index = pd.timedelta_range(start=0, periods=df.shape[0], freq=sample_freq)
    df = df.set_index(time_index)

    if target_freq is not None:
        target_freq = str(1/target_freq * 1E9) + 'N'
        df = df.resample(target_freq).mean()
    print(f"{session} was imported and resampled.")

    seizures = [d for d in summary if d.get('File Name') == session]
    df['is_seizure'] = 0
    for seizure in seizures:
        start = str(seizure['Seizure Start Time'] - 1) + "S"
        end = str(seizure['Seizure End Time'] - 1) + "S"
        df.loc[start:end, 'is_seizure'] = 1
        print(f"{session} seizure was labeled")
    return df


#%% load edf

def import_patients(root_dir=DATA_ROOT, patient_ids=[1], target_freq=128):
    '''load concatenated data of specified patients into pandas dataframe with labeled seizures by a list of patient_ids.
    
    root_dir: root directory of data. default: "repository/data/"
    patient_ids: list of integers defining the patients. if None: all patients are imported. default: [1]
    target_freq: target freqency in Hz to return the dataframe

    returns: pandas dataframe
    '''

    patient_list = get_patient_list(patient_ids=patient_ids)
    df_patient_list = []
    for patient in patient_list:
        summary = get_patient_summary(patient)
        # load edf
        df_patient = pd.concat([return_pandas_df(patient=patient, session=s, target_freq=target_freq, summary=summary) for s in get_session_list(patient=patient)])
        print(f'patient {patient} sessions concatenated.')
        # new_index = df_patient_list[0].index.union([i.index for i in df_patient_list[1:]])
        df_patient_list.append(df_patient)

    patient_all = pd.concat(df_patient_list)
    
    # reapply freqency index after concatination
    freq_string = str(1/target_freq * 1E9) + 'N'
    new_index = pd.timedelta_range(start=0, periods=patient_all.shape[0], freq=freq_string)
    patient_all = patient_all.set_index(new_index)
    return patient_all

### todo:
    # save default (single/full?) dataframes as pyarrow for faster access to the dataframe with low memory consumption.

    ### function load_eeg_full: 
    ### function load_eeg_single


#%%
if __name__ == "__main__":
    assert get_patient_list(patient_ids=[1,2,5]) == ['chb01', 'chb02', 'chb05']
    assert get_patient_summary()[3]['Seizure End Time'] == 1066
    
    # print(get_session_list())

    patients = import_patients(patient_ids=[2,3,4,8], target_freq=32)


