import pandas as pd
from source import data_import
from source.calculate_mean_psd import calculate_mean_psd
from source.constants import CHANNELS, FREQUENCY_RANGES


def extract_features(df):

    df_pp = df

    # ignore for aggregation
    ignore_col = ['is_seizure']
    ignore_col = []

    ### aggregate Functions for mean psd:
    delta = lambda x: calculate_mean_psd(x, frequency_ranges={'Delta' : FREQUENCY_RANGES['Delta']})[x.name]['Delta']
    theta = lambda x: calculate_mean_psd(x, frequency_ranges={'Theta' : FREQUENCY_RANGES['Theta']})[x.name]['Theta']
    gamma = lambda x: calculate_mean_psd(x, frequency_ranges={'Gamma': FREQUENCY_RANGES['Gamma']})[x.name]['Gamma']

    delta_agg = pd.NamedAgg(column='delta', aggfunc=delta)
    theta_agg = pd.NamedAgg(column='theta', aggfunc=theta)
    gamma_agg = pd.NamedAgg(column='gamma', aggfunc=gamma)
    abs_mean = lambda x: x.apply(abs).mean()
    abs_mean_agg = pd.NamedAgg(column='abs_mean', aggfunc=abs_mean)


    df_features = df_pp.groupby(['seizure_id', 'segment_id', "is_seizure"]).agg(
        {C:[
            # 'mean', 
            'std',
            'var',
            'mean',
            abs_mean_agg,
            delta_agg,
            theta_agg,
            gamma_agg
            ] for C in CHANNELS} | 
        {ic: ['first'] for ic in ignore_col}) # just taking first element for target column
    
    # joining column names with agg functions, but leaving target 'is_seizure' column as 'is_seizure'.
    df_features.columns = ['_'.join(col).strip() for col in df_features.columns.values if 'is_seizure' != col[0]] #+ ['is_seizure']
    df_features.reset_index(inplace=True)
    
    extracted_features = df_features 


    return extracted_features