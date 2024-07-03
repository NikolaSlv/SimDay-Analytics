import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

def load_dtype(file_path):
    '''
    Load the data types for the columns from a file.
    
    Parameters:
    file_path (str): The path to the file containing the data types.

    Returns:
    dict: A dictionary containing the data types for the columns.
    '''
    dtype = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, val = line.strip().split(':')
            if val == 'str':
                dtype[key] = str
            elif val == 'float':
                dtype[key] = float
    return dtype

def add_segment_to_dict(df, segments, current_segment, current_utc):
    '''
    Add the current segment to the dictionary of segments.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    segments (dict): The dictionary of segments.
    current_segment (list): The current segment.
    current_utc (str): The current UTC time.

    Returns:
    None
    '''
    segment_df = pd.DataFrame(current_segment, columns=df.columns)
    segment_df = segment_df.sort_values(by='WhoAmI')
    matrix = segment_df.drop(columns=['UTCISO8601', 'WhoAmI']).to_numpy()

    segments[current_utc] = matrix

def split_data_into_segments(df):
    '''
    Split the data into segments based on the UTC time.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    
    Returns:
    dict: A dictionary containing the segments.
    '''
    segments = {}
    current_segment = []
    current_utc = None

    for idx, row in df.iterrows():
        print(f'Processing row {idx}/{df.shape[0]}', end='\r')
        if not pd.isna(row['UTCISO8601']) and row['UTCISO8601'] != current_utc:
            if current_segment:
                add_segment_to_dict(df, segments, current_segment, current_utc)
            current_utc = row['UTCISO8601']
            current_segment = []
        current_segment.append(row)

    if current_segment:
        add_segment_to_dict(df, segments, current_segment, current_utc)

    return segments

def concatenate_hours(segments, normalize):
    '''
    Concatenate the segments into days and normalize each day's matrices column by column.

    Parameters:
    segments (dict): A dictionary containing the segments.

    Returns:
    dict: A dictionary containing the normalized daily matrices.
    '''
    days = {}
    for key in segments:
        day = key.split('T')[0]
        if day not in days:
            days[day] = {}
            days[day]['mat1'] = segments[key][:, :2]
            days[day]['mat2'] = segments[key][:, 2:5]
            days[day]['mat3'] = np.concatenate((segments[key][:, 5:6], segments[key][:, 7:]), axis=1)
        else:
            days[day]['mat1'] = np.concatenate((days[day]['mat1'], segments[key][:, :2]), axis=0)
            days[day]['mat2'] = np.concatenate((days[day]['mat2'], segments[key][:, 2:5]), axis=0)
            days[day]['mat3'] = np.concatenate((days[day]['mat3'], np.concatenate((segments[key][:, 5:6], segments[key][:, 7:]), axis=1)), axis=0)

    return days

def save_matrices(year, month, days, output_path):
    '''
    Save the matrices to the output path as numpy files.

    Parameters:
    matrices (dict): A dictionary containing the matrices.

    Returns:
    None
    '''
    output_path = output_path + year + '/' + month + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for key in days:
        np.save(output_path + f'mat1_{key}.npy', days[key]['mat1'])
        np.save(output_path + f'mat2_{key}.npy', days[key]['mat2'])
        np.save(output_path + f'mat3_{key}.npy', days[key]['mat3'])

def process_file(file_path, dtype, normalize, save):
    '''
    Processes a single file.

    Parameters:
    file_path (str): The path to the file.
    dtype (dict): A dictionary containing the data types for the columns.

    Returns:
    None
    '''
    try:
        df = pd.read_csv(file_path, dtype=dtype)
    except pd.errors.ParserError as e:
        print(f"Error processing {file_path}: {e}")
        return None
    df = df.iloc[:, 1:]  # Remove the first column
    print(f"File loaded: {file_path}, shape: {df.shape}")

    # Normalize the data (last 8 columns)
    df.iloc[:, -8:] = df.iloc[:, -8:].apply(lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x, axis=0)
    # df.iloc[:, -8:] = df.iloc[:, -8:].apply(lambda x: (x - x.mean()) / x.std(), axis=0) - another method of normalization, but worse MAPE results

    segments = split_data_into_segments(df)
    days = concatenate_hours(segments, normalize)

    if save:
        year_month = file_path.split('/')[-1].split('.')[0]
        year = year_month[:4]
        month = year_month[4:]
        save_matrices(year, month, days, './output/')
    else:
        return days

def run_parallel(file_paths, normalize=True, save=True):
    print(f'Running gen_matrices on {file_paths}')
    start_time = time.time()

    dtype = load_dtype('./config/dtype.txt')

    res = None
    with Pool(min(cpu_count(), 61)) as pool:
        res = pool.starmap(process_file, [(file_path, dtype, normalize, save) for file_path in file_paths])

    all_dict = None
    if res is not None:
        all_dict = {key: value for days in res if days is not None for key, value in days.items()}

    elapsed_time = time.time() - start_time
    print(f"Took {elapsed_time//86400} days, {elapsed_time//3600%24} hrs, {elapsed_time//60%60} mins, {elapsed_time%60:.2f} secs")

    if all_dict is not None:
        return all_dict
