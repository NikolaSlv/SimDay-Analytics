import pandas as pd
import numpy as np
import os
import time

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

def normalize_matrix_columns(matrix):
    '''
    Normalize each column of the matrix.

    Parameters:
    matrix (ndarray): The matrix to normalize.

    Returns:
    ndarray: The normalized matrix.
    '''
    for i in range(matrix.shape[1]):
        column = matrix[:, i]
        norm = np.linalg.norm(column)
        if norm != 0:
            matrix[:, i] = column / norm
    return matrix

def concatenate_hours(segments):
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
            # For the wind conditions matrix, exclude the wind direction column
            days[day]['mat3'] = np.concatenate((segments[key][:, 5:6], segments[key][:, 7:]), axis=1)
        else:
            days[day]['mat1'] = np.concatenate((days[day]['mat1'], segments[key][:, :2]), axis=0)
            days[day]['mat2'] = np.concatenate((days[day]['mat2'], segments[key][:, 2:5]), axis=0)
            days[day]['mat3'] = np.concatenate((days[day]['mat3'], np.concatenate((segments[key][:, 5:6], segments[key][:, 7:]), axis=1)), axis=0)
    
    # Normalize each day's matrices column by column
    for day in days:
        days[day]['mat1'] = normalize_matrix_columns(days[day]['mat1'])
        days[day]['mat2'] = normalize_matrix_columns(days[day]['mat2'])
        days[day]['mat3'] = normalize_matrix_columns(days[day]['mat3'])

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

def load_file(file_path, dtype):
    '''
    Loads and processes the data from a file.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    None
    '''
    df = pd.read_csv(file_path, dtype=dtype)
    df = df.iloc[:, 1:] # Remove the first column
    print(f"File loaded: {file_path}, shape: {df.shape}")

    segments = split_data_into_segments(df)
    days = concatenate_hours(segments)

    year_month = file_path.split('/')[-1].split('.')[0]
    year = year_month[:4]
    month = year_month[4:]
    save_matrices(year, month, days, './output/')

def run(input):
    print(f'Running gen_matrices on {input}')
    start_time = time.time()

    dtype = load_dtype('./config/dtype.txt')
    load_file(input, dtype)

    elapsed_time = time.time() - start_time
    print(f"Took {elapsed_time//86400} days, {elapsed_time//3600%24} hrs, {elapsed_time//60%60} mins, {elapsed_time%60:.2f} secs")
