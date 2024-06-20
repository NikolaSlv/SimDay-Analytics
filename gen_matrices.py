import pandas as pd
import numpy as np
import os

'''
Global variables:
dtype (dict): A dictionary containing the data types for the columns.
'''
dtype = None

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

    # matrix = segment_df.drop(columns=['UTCISO8601', 'WhoAmI']).to_numpy()
    matrix = segment_df.drop(columns=['UTCISO8601', 'WhoAmI']).head(30).to_numpy() # FOR TESTING

    # Iterate through each column and normalize it
    for i in range(matrix.shape[1]):
        column = matrix[:, i]
        norm = np.linalg.norm(column)
        if norm == 0:
            column = np.zeros_like(column)
        else:
            column = column / norm
        matrix[:, i] = column

    segments[current_utc] = matrix
    print(f'UTC time: {current_utc}, shape: {matrix.shape}, successfully loaded.')

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
        if idx > 1500000:
            break
        if pd.notna(row['UTCISO8601']):
            if current_segment:
                add_segment_to_dict(df, segments, current_segment, current_utc)
            current_utc = row['UTCISO8601']
            current_segment = []
        current_segment.append(row)

    if current_segment:
        add_segment_to_dict(df, segments, current_segment, current_utc)

    return segments

def concatinate_hours(segments):
    '''
    Concatinate the segments into days. Each segment represents an hour.

    Parameters:
    segments (dict): A dictionary containing the segments.

    Returns:
    dict: A dictionary containing the days.
    '''
    days = {}
    for key in segments:
        day = key.split('T')[0]
        if day not in days:
            days[day] = {}
            days[day]['mat1'] = segments[key][:, :2]
            days[day]['mat2'] = segments[key][:, 2:5]
            days[day]['mat3'] = segments[key][:, 5:]
        else:
            days[day]['mat1'] = np.concatenate((days[day]['mat1'], segments[key][:, :2]), axis=0)
            days[day]['mat2'] = np.concatenate((days[day]['mat2'], segments[key][:, 2:5]), axis=0)
            days[day]['mat3'] = np.concatenate((days[day]['mat3'], segments[key][:, 5:]), axis=0)
    return days

def save_matrices(year_month, days, output_path):
    '''
    Save the matrices to the output path as numpy files.

    Parameters:
    matrices (dict): A dictionary containing the matrices.

    Returns:
    None
    '''
    output_path = output_path + year_month + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for key in days:
        np.save(output_path + f'mat1_{key}.npy', days[key]['mat1'])
        np.save(output_path + f'mat2_{key}.npy', days[key]['mat2'])
        np.save(output_path + f'mat3_{key}.npy', days[key]['mat3'])
        print(f'Matrices for {key} saved.')

def load_file(file_path):
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
    days = concatinate_hours(segments)

    year_month = file_path.split('/')[-1].split('.')[0]
    save_matrices(year_month, days, './output/')

def main():
    dtype = load_dtype('./dtype.txt')
    load_file('./data/194001.csv')

if __name__ == '__main__':
    main()
