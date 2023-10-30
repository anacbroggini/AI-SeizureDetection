'''Some tools to import .edf patient sessions. 

execution functions: 
    import_patients(root_dir, patient_ids, target_freq)
        load concatenated edf data of specified patients into pandas dataframe with labeled seizures by a list of patient_ids.
        input:
            root_dir: root directory of data. default: "repository/data/"
            patient_ids: list of integers defining the patients. if None: all patients are imported. default: [1]
            target_freq: target freqency in Hz to return the dataframe
        output:
            returns: pandas dataframe

'''


#%%

from pathlib import Path
import numpy as np
import pyarrow as pa
import pandas as pd
# import modin.pandas as pd
import mne

from time import time 
import warnings

DATA_ROOT = Path(__file__).resolve().parent / "../data"
EEG_LARGE_FILENAME = "eeg_large.arrow"
EEG_SINGLE_FILENAME = "eeg_single.arrow"

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
    # df = df.copy().astype('double[pyarrow]')
    # df = df.copy().astype(pd.ArrowDtype(pa.float64()))

    if target_freq != 256:
        target_freq = str(1/target_freq * 1E9) + 'N'
        df = df.resample(target_freq).mean()
        print(f"{session} was imported and resampled.")
    else:
        print(f"{session} was import but not resampled {target_freq}Hz.")

    seizures = [d for d in summary if d.get('File Name') == session]
    df['is_seizure'] = False
    # df['is_seizure'] = df['is_seizure'].astype(pd.ArrowDtype(pa.bool_()))
    for seizure in seizures:
        start = str(seizure['Seizure Start Time'] - 1) + "S"
        end = str(seizure['Seizure End Time'] - 1) + "S"
        df.loc[start:end, 'is_seizure'] = True
        print(f"{session} seizure was labeled")

    return df #.astype(pd.ArrowDtype(pa.float64()))


#%% load edf

def import_patients(root_dir=DATA_ROOT, patient_ids=[1], target_freq=256):
    '''load concatenated edf data of specified patients into pandas dataframe with labeled seizures by a list of patient_ids.
    
    root_dir: root directory of data. default: "repository/data/"
    patient_ids: list of integers defining the patients. if None: all patients are imported. default: [1]
    target_freq: target freqency in Hz to return the dataframe. If None no resampling is done.

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
    print(f'data of patients {patient_ids} were concatenated.')
    
    # reapply freqency index after concatination
    freq_string = str(1/target_freq * 1E9) + 'N'
    new_index = pd.timedelta_range(start=0, periods=patient_all.shape[0], freq=freq_string)
    patient_all = patient_all.set_index(new_index)
    return patient_all #.astype(pd.ArrowDtype(pa.float64()))


def save_pyarrow(data=None, path_name=DATA_ROOT, file_name='pyarrow_df'):
    '''save dataframe to disc, which can then also loaded by memory-mapping.'''
    
    if not file_name.endswith('.arrow'):
        file_name += '.arrow'
    arrow_filepath = Path(path_name) / file_name

    table = pa.Table.from_pandas(data, preserve_index=True)
    with pa.OSFile(str(arrow_filepath), 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    print(f"{str(arrow_filepath)} was successfully written.")
    return

def load_pyarrow(path_name=DATA_ROOT, file_name='pyarrow_df'):
    '''load dataframe from disc via memory mapping (basically no ram-memory is used.)'''

    if not file_name.endswith('.arrow'):
        file_name += '.arrow'
    arrow_filepath = Path(path_name) / file_name

    try:
        source = pa.memory_map(str(arrow_filepath), 'r')
        print(f"{str(arrow_filepath)} was loaded.")
        arrow_patient_df = pa.ipc.RecordBatchFileReader(source).read_pandas()
        return arrow_patient_df
    except FileNotFoundError as e:
        warnings.warn(f"FileNotFoundError: {e}")
        return

# helper functions to make life easier. save and load predefined datasets (large & single)
def load_eeg_large_mem():
    '''load a predefined eeg data (called large) as memory map with py_arrow.'''
    
    try:
        df = load_pyarrow(file_name=EEG_LARGE_FILENAME)
        return df
    except FileNotFoundError as e:
        warnings.warn(f"FileNotFoundError: {e}")
    return 

def load_eeg_single_mem():
    '''load a predefined eeg data of a single patient as memory map with py_arrow.'''
    try:
        df = load_pyarrow(file_name=EEG_SINGLE_FILENAME)
        return df
    except FileNotFoundError as e:
        warnings.warn(f"FileNotFoundError: {e}")
    return 

def save_pyarrow_eeg_large(data=None, patient_ids=[1,2,3,12]):
    """save a default large data to predefined path."""
    
    if data is None:
        data = import_patients(patient_ids=[patient_ids])
    
    save_pyarrow(data, file_name=EEG_LARGE_FILENAME)
    return

def save_pyarrow_eeg_single(data=None, patient_id=12):
    """save a default dataset of single patient. 
    
    input:
        data - optional: dataset to be saved.
        patient_id - instead of data, a patient_id can be given which dataset will be loaded from edf first instead.
    """
    
    if data is None:
        data = import_patients(patient_ids=[patient_id])
    
    print(data.shape)

    save_pyarrow(data, file_name=EEG_SINGLE_FILENAME)
    return


#%%
if __name__ == "__main__":
    assert get_patient_list(patient_ids=[1,2,5]) == ['chb01', 'chb02', 'chb05']
    assert get_patient_summary()[3]['Seizure End Time'] == 1066
    
    patients = import_patients(patient_ids=[11])
    print("single patient done.")
    save_pyarrow_eeg_single(patients)
    # print(get_session_list())

    patients = import_patients(patient_ids=[1,2,3,12], target_freq=32)
    save_pyarrow_eeg_large(patients)
    tic = time()
    patients.mean()
    print(time()- tic)
    del patients

    df = load_eeg_large_mem()

    tic = time()
    df.mean()
    print(f"{time()- tic}")

