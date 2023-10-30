import os
import pandas as pd
import numpy as np

def get_seizure_sequence(source_path="data"):
    file_names = []
    number_of_seizures = []
    seizure_start = []
    seizure_end = []

    for folder_name in os.listdir(source_path):
        if not folder_name.startswith("."):
            folder_path = os.path.join(source_path, folder_name)
                
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".txt"):
                        file_path = os.path.join(folder_path, file_name)

            with open(file_path) as f:

                for line in f:
                    line = line.strip()

                    if "File" in line and "Name" in line:
                        file_names.append(line.split(": ", 1)[1])

                    if "Number" in line and "Seizure" in line:
                        number_of_seizures.append(int(line.split(": ", 1)[1]))

                    if "Seizure" in line and "Start" in line:
                        seizure_start.append(float(line.split(": ", 1)[1].split()[0]))

                    if "Seizure" in line and "End" in line:
                        seizure_end.append(float(line.split(": ", 1)[1].split()[0]))

    data = {
        "file_name": file_names,
        "number_of_seizures": number_of_seizures
    }

    df_seizure = pd.DataFrame(data)
    df_seizures_tmp = df_seizure.loc[df_seizure.index.repeat(df_seizure['number_of_seizures'])]
    df_no_seizures_tmp = df_seizure[df_seizure["number_of_seizures"] == 0]

    df = pd.concat([df_seizures_tmp, df_no_seizures_tmp], ignore_index=True)
    df["seizure_start"] = np.where(df["number_of_seizures"] == 0, np.nan, seizure_start * (len(df) // len(seizure_start)) + seizure_start[:len(df) % len(seizure_start)])
    df["seizure_end"] = np.where(df["number_of_seizures"] == 0, np.nan, seizure_end * (len(df) // len(seizure_end)) + seizure_end[:len(df) % len(seizure_end)])

    return df

def get_seizure_mask(source_path="data"):
    file_names = []
    seizure_mask = []


    for folder_name in os.listdir(source_path):
        if not folder_name.startswith("."):
            folder_path = os.path.join(source_path, folder_name)
                
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".txt"):
                        file_path = os.path.join(folder_path, file_name)

            with open(file_path) as f:

                for line in f:
                    line = line.strip()

                    if "File" in line and "Name" in line:
                        file_names.append(line.split(": ", 1)[1])

                    if "Number" in line and "Seizure" in line:
                        if int(line.split(": ", 1)[1]) == 0:
                            seizure_mask.append(False)
                        else:
                            seizure_mask.append(True)

    data = {
        "file_name": file_names,
        "seizure_mask": seizure_mask
    }

    df = pd.DataFrame(data)
    
    return df

def assign_seizures(edf_df, time_var, seizures_df):
    edf_df["seizure_sequence"] = False
    
    for index, row in edf_df.iterrows():
        file_name = row["file_name"]
        time = row[time_var]
        matching_seizure = seizures_df[(seizures_df["file_name"] == file_name) & (seizures_df["seizure_start"] <= time) & (seizures_df["seizure_end"] >= time)]
        
        if not matching_seizure.empty:
            edf_df.at[index, "seizure_sequence"] = True
    
    return edf_df