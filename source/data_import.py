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
from matplotlib.artist import get
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
def get_session_list(root_dir=DATA_ROOT, patient='chb01', seizure_flag=None):
    '''get filenames of sessions for specific patient, ie. ["chb01_01.edf", ...] '''

    root_dir = Path(root_dir)
    if seizure_flag is True:
        # filter only edf files with seizures
        session_list = sorted([s.name for s in (root_dir / patient).rglob(pattern='*.seizures')])
        session_list = [s.rstrip('.seizure') for s in session_list]
    elif seizure_flag is False:
        # filter only edf files without seizures
        exclude_list = sorted([s.name for s in (root_dir / patient).rglob(pattern='*.seizures')])
        exclude_list = [s.rstrip('.seizure') for s in exclude_list]
        session_list = sorted([s.name for s in (root_dir / patient).rglob('*.edf') if s.name not in exclude_list])
    elif seizure_flag is None:
        # get all edf files
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
                    "file_name": None, 
                    "number_of_seizures": None, 
                    "seizure_start_time": np.nan, 
                    "seizure_end_time": np.nan, 
                    "patient": patient
                    }
                current_data["file_name"] = line.split(": ", 1)[1]
            
            elif line.startswith("Number of Seizures in File:"):
                current_data["number_of_seizures"] = line.split(": ", 1)[1]
                if current_data["number_of_seizures"] == 0:
                    data.append(current_data.copy())
            
            # elif line.startswith("seizure_start_time:"):
            elif ("Seizure" in line) and ("Start" in line):
                current_data["seizure_start_time"] = int(line.split(": ", 1)[1].split()[0])
            
            # elif line.startswith("seizure_end_time:"):
            elif "Seizure" in line and "End" in line:
                current_data["seizure_end_time"] = int(line.split(": ", 1)[1].split()[0])
                data.append(current_data.copy())

    return data


### function import_eeg_data
#%%
def return_pandas_df(root_dir=DATA_ROOT, patient=None, session=None, target_freq=256, summary=None, channels=None, preseizure_interval=300):
    ''' Read specific session .edf, transform to pandas dataframe with timedelta index, optional resampling, labeling seizures in "is_seizure" column.
    
    returns pandas dataframe of session.
    '''
    
    if channels is None:
        ch = None
    else:
        ch = channels.copy()
        if 'T8-P8-1' in ch:
            ch.remove('T8-P8-1')
            ch.append('T8-P8')
        if 'T8-P8-0' in ch:
            ch.remove('T8-P8-0')
            ch.append('T8-P8')
    
    file_path = root_dir / patient / session
        
    raw = mne.io.read_raw_edf(file_path, preload=False, include=ch, verbose='ERROR')
    
    sample_freq = str(1/raw.info['sfreq'] * 1E9) + 'N'
    try:
        df = raw.to_data_frame(index='time')
    except ValueError:
        print(f"couldnt load datafile {patient + session}")
        return None,None
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
    
    df['is_seizure'] = False
    df['before_seizure'] = False
    # df['file'] = patient + '/' + session
    # df['patient_id'] = str([s for s in patient if s.isdigit()])
    # df['file_id'] = str([s for s in session if s.isdigit()])
    seizures = [d for d in summary if d.get('file_name') == session]
    # df['is_seizure'] = df['is_seizure'].astype(pd.ArrowDtype(pa.bool_()))
    for seizure in seizures:
        # label seizure
        start = str(seizure["seizure_start_time"] - 1) + "S"
        end = str(seizure['seizure_end_time'] - 1) + "S"
        df.loc[start:end, 'is_seizure'] = True

        # label buffer
        buffer_length = preseizure_interval  # buffer length in seconds
        start = str(max(0, seizure["seizure_start_time"] - 1 - buffer_length)) + "S"
        end = str(min(df.index[-1].seconds, seizure['seizure_start_time'] - 2)) + "S"
        df.loc[start:end, 'before_seizure'] = True
        print(f"{session} seizure and buffer was labeled")

    return df, bool(len(seizures)) #.astype(pd.ArrowDtype(pa.float64()))


#%% load edf

def import_patients(root_dir=DATA_ROOT, patient_ids=[1], target_freq=256, seizure_flag=None, channels=None, preseizure_interval=300):
    '''load concatenated edf data of specified patients into pandas dataframe with labeled seizures by a list of patient_ids.
    
    root_dir: root directory of data. default: "repository/data/"
    patient_ids: list of integers defining the patients. if None: all patients are imported. default: [1]
    target_freq: target freqency in Hz to return the dataframe. If None no resampling is done.
    seizure_flag: True/False/None. True return all content of 

    returns: pandas dataframe
    '''

    patient_list = get_patient_list(patient_ids=patient_ids)
    df_patient_list = []
    for patient in patient_list:

        summary = get_patient_summary(patient=patient)
        # load edf
        df_patient = pd.concat([
            return_pandas_df(patient=patient, 
                             session=s, 
                             target_freq=target_freq, 
                             summary=summary,
                             channels=channels,
                             preseizure_interval=preseizure_interval,
                             )[0] for s in get_session_list(patient=patient, seizure_flag=seizure_flag)])
        
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


### Add Segmentation functions to the loading of data

def add_segmentation_report(report, key, value):
        report[key].append(value)
        return report


def ictal_segmentation(df, epoch=0, duration_segment=10, nr_segments = 20, **kwargs):
    '''segment for ictal intervals. it adds as many ictal segments as available and fills them up with pre-ictal segments to reach the total nr of segments.
    
    if end of data is reached before the end of seizure, only this part of seizure will be used.
    
    '''

    global segmentation_report

    # how to validate nr_segments is well chosen? 
    df['segment_id'] = 0
    ictal_epochs = []
 
    # for time, seizure in (s_df[s_df == True]).items():
    for ep_start, seizure in df.loc[df['seizure_start'] == True, 'seizure_start'].items():
        ictal_segments = []
        segment_id = 1
        seg_start = ep_start
        while True:
            if segment_id > 0.66 * nr_segments:
                print(f"reached maximum ratio for seizure segments ({segment_id - 1}). skipping further segments ...")
                segmentation_report = add_segmentation_report(
                    segmentation_report, 'seizure_max_reached', [epoch, seg_start])
                break

            # get previous duration_segment duration in seconds
            seg_end = seg_start + (pd.Timedelta(seconds=duration_segment)) - df.index.freq
            if seg_end > df.index[-1]:
                print(f"reached end of data. epoch number: {epoch}, number of segments completed: {segment_id-1}")
                segmentation_report = add_segmentation_report(
                    segmentation_report, 'reached_file_end', [epoch, seg_start])
                break
            
            ictal_seg = df.loc[seg_start: seg_end, :].copy()
            ictal_seg['segment_id'] = segment_id
        
            # check if is_seizure in it AND if all are before_seizure
            # s_int.loc['10S', 'is_seizure'] = True  # check
            if not all(ictal_seg['is_seizure']):
                print(f"seizure end reached. epoch number: {epoch}, number of segments completed: {segment_id-1}")
                break # get out of while loop
            
            ictal_segments.append(ictal_seg)
            seg_start =  seg_end + df.index.freq
            segment_id += 1
        
        print(f"adding {segment_id-1} ictal segments to epoch {epoch}.") # segment_id is already incremented, but starts with 0: x-1+1=x
        if len(ictal_segments) == 0:
            print("first ictal segment is too small.")
            segmentation_report = add_segmentation_report(
                segmentation_report, 'seizure_too_small', [epoch, seg_start])
            continue
        else:
            ictal_epoch = pd.concat(ictal_segments)
        
        ### get missing (pre-ictal) segments before ictal
        segments_missing = nr_segments - (segment_id - 1)
        # calc start / end: 
        preictal_start = ep_start - segments_missing * pd.Timedelta(seconds=duration_segment)
        preictal_end = ep_start - df.index.freq

        # validate target epoch is within dataframe
        if preictal_start < df.index[0]:
            print(f"preictal interval is not fully covered by datafile! skipping ...")
            segmentation_report = add_segmentation_report(
                segmentation_report, 'reached_file_start', [epoch, seg_start])
            continue
        # get_signal: 
        preictal_ep = df.loc[preictal_start:preictal_end, :].copy()
        # validate signal is not ictal
        if any(preictal_ep['is_seizure']):
            print(f"overlapping ictal interval for this episode. skipping ...")
            continue
        
        # merge with ictal part of interval
        print(f"adding {segments_missing} pre-ictal segments to epoch {epoch}.")
        # full_epoch = preictal_ep.join(ictal_epoch)
        full_epoch = pd.concat([preictal_ep, ictal_epoch])
        # set segment_ids
        full_epoch['segment_id'] = [i for i in range(nr_segments) for _ in range(int(len(full_epoch)/nr_segments))]
        full_epoch['epoch'] = epoch
        epoch += 1

        # add full epoch to epoch list
        ictal_epochs.append(full_epoch)
        segmentation_report = add_segmentation_report(
            segmentation_report, 'ictal_epochs', [epoch, seg_start])
    
    return ictal_epochs


#%%
def preictal_segmentation(df, epoch=0, duration_segment=10, nr_segments=20, seizure_offset=0):
    """ get segments purely from preictal intervals.

    Args:
        df: pandas dataframe of patient session (whole file).
        epoch: current epoch number. Defaults to 0.
        duration_segment: length of segment in seconds. Defaults to 10.
        nr_segments: number of segments per sequence. Defaults to 20.
        seizure_offset: buffer interval between end of sequence and seizure start. Defaults to 0.

    Returns:
        returns a list of preictal epochs for current session-file.
        columns: ['is_seizure', 'before_seizure', 'segment_id', 'target', 'epoch']
            'target': seconds from sequence end until seizure start
    """    
        
    global segmentation_report

    # how to validate nr_segments is well chosen? 
    df['segment_id'] = 0
    preictal_epochs = []
 
    # for time, seizure in (s_df[s_df == True]).items():
    for ep_start, seizure in df.loc[df['seizure_start'] == True, 'seizure_start'].items():
        ep_start -= pd.Timedelta(seconds=seizure_offset)
        preictal_segments = []

        train_length = duration_segment * nr_segments
        seg_start = ep_start - pd.Timedelta(seconds=train_length)
        seg_end = ep_start - df.index.freq
        if seg_start < df.index[0]:
            print(f"preictal interval is not fully covered by datafile! skipping ...")
            segmentation_report = add_segmentation_report(
                segmentation_report, 'reached_file_start', [epoch, seg_start])
            continue
        preictal_epoch = df.loc[seg_start: seg_end, :].copy()
        if any(preictal_epoch['is_seizure']):
            print(f"overlapping ictal interval for this episode. skipping ...")
            segmentation_report = add_segmentation_report(
                segmentation_report, 'overlapping_seizure', [epoch, seg_start])
            continue
        preictal_epoch['segment_id'] = [i for i in range(nr_segments) for _ in range(int(len(preictal_epoch)/nr_segments))] 
        preictal_epoch['epoch'] = epoch
        preictal_epoch['target'] = ep_start - preictal_epoch.index + pd.Timedelta(seizure_offset)
        epoch += 1
        preictal_epochs.append(preictal_epoch)
        segmentation_report = add_segmentation_report(
            segmentation_report, 'preictal_epochs', [epoch, seg_start])
    return preictal_epochs



#%%
def inter_segmentation(df, epoch=0, duration_segment=10, nr_segments=20):
    '''segment function that just adds an "epoch" from the middle of a seizure free datafile.'''

    global segmentation_report    

    inter_ictal_buffer= pd.Timedelta(seconds=1200) # at least 20 minutes at beginning of file needed for making it interictal
    interictal_epoch = []

    # from 20 min in the dataframe, get the segments
    start = df.index[-1] // 2
    epoch_duration = pd.Timedelta(seconds = nr_segments * duration_segment)
    if (start < inter_ictal_buffer) or (start+epoch_duration > df.index[-1]):
        # check if file is too short for hosting a guarantueed interictal phase of specified length
        print("file too short to host interictal sequence.")
        segmentation_report = add_segmentation_report(segmentation_report, 'interictal_skipped', [epoch, start])
        return interictal_epoch
    segments = df.loc[start:start + epoch_duration, :].copy()
    # add segment numbers and epoch id
    segments['epoch'] = epoch
    segments['segment_id'] = [i for i in range(nr_segments) for _ in range(int(len(segments)/nr_segments))]
    segmentation_report = add_segmentation_report(
            segmentation_report, 'interictal_epochs', [epoch, start])
    interictal_epoch.append(segments)
    return interictal_epoch

def load_segmented_unlabeled_data(file_path, channels=None):
    """load edf-file and segment it to apply classification model."""
    segment_duration = 5


    # load edf into pandas df
    if channels is None:
        ch = None
    else:
        ch = channels.copy()
        if 'T8-P8-1' in ch:
            ch.remove('T8-P8-1')
            ch.append('T8-P8')
        if 'T8-P8-0' in ch:
            ch.remove('T8-P8-0')
            ch.append('T8-P8')
    
    file_path = Path(file_path)
    raw = mne.io.read_raw_edf(file_path, preload=False, include=ch, verbose='ERROR')
    sample_freq = str(1/raw.info['sfreq'] * 1E9) + 'N'
    df = raw.to_data_frame(index='time')
    time_index = pd.timedelta_range(start=0, periods=df.shape[0], freq=sample_freq)
    df = df.set_index(time_index)


    # do segmentation of whole file
    nr_segments = int(len(df.index) // (segment_duration * raw.info['sfreq']))
    crop_at = pd.Timedelta(seconds=nr_segments * segment_duration)
    segments = df.loc[:crop_at,:].copy()
    # add segment numbers and epoch id
    # segments['epoch'] = 0
    segments['segment_id'] = [i for i in range(nr_segments) for _ in range(int(len(segments)/nr_segments))]
    
    return  segments



def load_segmented_data(root_dir=DATA_ROOT, 
                        patient_ids=[1], 
                        target_freq=256,
                        nr_segments=15,
                        segment_duration=20,
                        seizure_offset=0,
                        ictal_segmentation_foo=ictal_segmentation,
                        interictal_segmentation_foo=inter_segmentation,
                        channels=None
                        ):
    """load data from patient edf files and segment it into epochs.

    Args:
        root_dir: path to patient data. Defaults to DATA_ROOT.
        patient_ids: integer ids of patients. Defaults to [1].
        target_freq: frequency of output . Defaults to 256.
        nr_segments: number of segments per epoch. Defaults to 15.
        segment_duration: length of segment in seconds. Defaults to 20.
        ictal_segmentation_foo: segmentation function for seizure positive sessions. Defaults to ictal_segmentation.
        interictal_segmentation_foo: segmentation function for seizure negative sessions. Defaults to inter_segmentation.
        channels: list of channels to be loaded. Defaults to None, which will load all channels.

    Returns:
        pandas dataframe of segmented data for all patients.
        columns: ['is_seizure', 'before_seizure', 'segment_id', 'target', 'epoch']
            'is_seizure': whether segment contains seizure
            'before_seizure': whether segment is within 300s before seizure
            'segment_id': number of segment within epoch
            'epoch': id of epoch
            'target': seconds from sequence end until seizure start

    """

    global segmentation_report
    segmentation_report = {
        'dataread_error': [],
        'reached_file_start': [], 
        'reached_file_end': [],
        'seizure_too_small': [],
        'overlapping_seizure': [],
        'ictal_epochs': [],
        'preictal_epochs': [],
        'interictal_epochs': [],
        'seizure_max_reached': [],
        'incomplete_channels': [],
        'interictal_skipped': []
        }
    
    patient_list = get_patient_list(patient_ids=patient_ids)

    epoch_counter = 0
    df_patients = []
    for patient in patient_list:

        summary = get_patient_summary(patient=patient)

        session_list = sorted([s.name for s in (root_dir / patient).rglob('*.edf')])
        session_dfs = []

        for session in session_list:
            df, is_seizure = return_pandas_df(patient=patient, session=session, summary=summary, channels=channels, target_freq=target_freq)
            if df is None:
                print("couldnt read data from edf ...")
                segmentation_report = add_segmentation_report(
                    segmentation_report, 'dataread_error', [patient, session])
                continue
            if channels is not None and [c for c in channels if c not in df.columns]:
                print("incomplete channels. skipping ...")
                segmentation_report = add_segmentation_report(
                    segmentation_report, 'incomplete_channels', [epoch_counter, session])
                continue
            df['seizure_start'] = df['is_seizure'] & ~df['is_seizure'].shift(fill_value=False)
            df['target'] = pd.Timedelta(seconds=0)
            if is_seizure:
                # session_dfs.append(ictal_segmentation(df, epoch = epoch_counter))
                session_dfs.extend(ictal_segmentation_foo(
                    df, 
                    epoch = epoch_counter, 
                    duration_segment=segment_duration, 
                    nr_segments=nr_segments,
                    seizure_offset=seizure_offset
                    )
                )
            else:
                session_dfs.extend(interictal_segmentation_foo(
                    df, 
                    epoch = epoch_counter, 
                    duration_segment=segment_duration, 
                    nr_segments=nr_segments
                    )
                )
            if len(session_dfs):
                epoch_counter = session_dfs[-1]['epoch'].max() + 1
        
        df_patients.extend(session_dfs)

    output = pd.concat(df_patients)

    [print(k,len(v)) for k,v in segmentation_report.items()]    
    return output


### get all seizure length

def get_seizures_per_patient(patient_ids):
    '''return a list of seizure lengths in seconds for all seizures of all given patients.'''
    patient_list = get_patient_list(patient_ids=patient_ids)
    seizures_per_patient = []
    for patient in patient_list:
        sessions = get_session_list()
        summary = get_patient_summary(patient=patient)
        seizures_per_patient.append(len(summary))
    return seizures_per_patient

def get_seizure_lengths(patient_ids):
    '''return a list of seizure lengths in seconds for all seizures of all given patients.'''
    patient_list = get_patient_list(patient_ids=patient_ids)
    seizure_lengths = []
    for patient in patient_list:
        sessions = get_session_list()
        summary = get_patient_summary(patient=patient)
        seizure_lengths.extend([seizure['seizure_end_time'] - seizure['seizure_start_time'] for seizure in summary if seizure['seizure_end_time'] != np.nan])
    return seizure_lengths
    

### Functions for saving and memory mapping dataframes
def save_pyarrow(data=None, path_name=DATA_ROOT, file_name='pyarrow_df'):
    '''save dataframe to disc, which can then also loaded by memory-mapping.'''
    
    if not file_name.endswith('.arrow'):
        file_name += '.arrow'
    arrow_filepath = Path(path_name) / file_name

    # table = pa.Table.from_pandas(data, preserve_index=True)
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
        if type(arrow_patient_df.index) is pd.core.indexes.timedeltas.TimedeltaIndex:
            freq = pd.infer_freq(arrow_patient_df.index)
            arrow_patient_df.index.freq = pd.tseries.frequencies.to_offset(freq)

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

def save_pyarrow_eeg_large(data=None, patient_ids=[1,2,3,4]):

    """save a default large data to predefined path."""
    
    if data is None:
        data = import_patients(patient_ids=patient_ids)
    
    save_pyarrow(data, file_name=EEG_LARGE_FILENAME)
    return


def save_pyarrow_eeg_single(data=None, patient_id=[1,2,3,4]):

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


def load_file(file_name):
    """convenience function for loading npy/arrow.

    Args:
        file_name: _description_

    Returns:
        _description_
    """
    if file_name.endswith('.arrow'):
        return load_pyarrow(file_name=file_name)
    elif file_name.endswith('.npy'):
        print('loading npy')
        return np.load('data/' + file_name, allow_pickle=True)
    else:
        print('no filename provided, trying npy')
        try:
            np.load('data/' + file_name + '.npy', allow_pickle=True)
        except FileNotFoundError:
            print('no npy file found, trying arrow')
            return load_pyarrow(file_name=file_name + '.arrow')
        
def save_file(data, file_name):
    """convenience function for saving npy/arrow.

    Args:
        data: _description_
        file_name: _description_
    """
    if file_name is None:
        print('skipping save file.')
    elif file_name.endswith('.arrow'):
        save_pyarrow(data, file_name=file_name)
    elif file_name.endswith('.npy'):
        print('saving npy')
        np.save('data/' + file_name, data)
    else:
        print('no filetype provided, saving as npy')
        np.save('data/' + file_name + '.npy', data)


#%%
if __name__ == "__main__":

    from constants import DEFAULT_PATIENTS, CHANNELS


    load_segmented_unlabeled_data('./data/chb01/chb01_03.edf', channels=CHANNELS)

    
    # # test get_patient_list
    # assert get_patient_list(patient_ids=[1,2,3,4]) == ['chb01', 'chb02', 'chb03', 'chb04']
    # # test get_patient_summary
    # assert get_patient_summary()[3]['seizure_end_time'] == 1066

    # # test segmented data retrieval
    # nr_segments = 20
    # segment_duration = 10
    # freq = 256
    # target_freq = 1
    # output = load_segmented_data(patient_ids=[17], 
    #                              nr_segments=nr_segments, 
    #                              segment_duration=segment_duration,
    #                              channels=CHANNELS,
    #                              target_freq=target_freq)
    # assert output[['epoch']].value_counts()[0] == segment_duration * nr_segments * freq * target_freq/freq
    # assert all(element == output[['epoch']].value_counts()[0] for element in output[['epoch']].value_counts())
    # assert all(element == (output['epoch'].max() + 1) * segment_duration * freq * target_freq/freq for element in output['segment_id'].value_counts())

    # output = load_segmented_data(patient_ids=[2], 
    #                              ictal_segmentation_foo=preictal_segmentation,
    #                              interictal_segmentation_foo=inter_segmentation,
    #                              nr_segments=nr_segments, segment_duration=segment_duration)
    # assert output['is_seizure'].sum() == 0
    
    # # patients = import_patients(patient_ids=[1,2,3,4], target_freq=32, seizure_flag=True)
    # # print(patients.shape)


    # # save_pyarrow_eeg_large(patient_ids=[1,2,3,4])

    
    # # patients = import_patients(patient_ids=[3])
    # # save_pyarrow_eeg_single()

    # # df = load_eeg_single_mem()
    # # print(df.mean())

    # # print("single patient done.")
    # # print("done.")
    # # # print(get_session_list())

    # # patients = import_patients(patient_ids=[1,2,3,12], target_freq=32)
    # # save_pyarrow_eeg_large(patients)
    # # tic = time()
    # # patients.mean()
    # # print(time()- tic)
    # # del patients

    # # df = load_eeg_large_mem()

    # # tic = time()
    # # df.mean()
    # # print(f"{time()- tic}")

