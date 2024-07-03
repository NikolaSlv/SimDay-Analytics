import pandas as pd
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

def create_set_for_tx(data):
    '''
    Create a set of concatenated longitude and latitude values for rows where the region is TX.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.

    Returns:
    set: A set of concatenated longitude and latitude values.
    '''
    tx_set = set()
    tx_data = data[data['Region'] == 'TX']
    for _, row in tx_data.iterrows():
        lon_lat = row['WhoAmI']
        tx_set.add(lon_lat)
    return tx_set

def filter_data(file_path, tx_set):
    '''
    Filter the data based on the longitude and latitude values in the file path.

    Parameters:
    file_path (str): The path to the file containing the data.
    tx_set (set): The set of concatenated longitude and latitude values.

    Returns:
    None
    '''
    year = os.path.basename(file_path)[:4]
    os.makedirs(f'./data_filtered/{year}', exist_ok=True) # Using exist_ok=True to avoid FileExistsError
    output_path = f'./data_filtered/{year}/{os.path.basename(file_path)}'

    header = True
    with open(output_path, 'w') as out:
        with open(file_path, 'r') as file:
            for line in file:
                if line.split(',')[2] in tx_set or header:
                    out.write(line)
                    header = False

def process_file(file_path, tx_set):
    '''
    Process a single file by filtering based on the tx_set.

    Parameters:
    file_path (str): The path to the file.
    tx_set (set): The set of concatenated longitude and latitude values.

    Returns:
    None
    '''
    filter_data(file_path, tx_set)

def run_parallel(file_paths):
    '''
    Run the processing in parallel for multiple files.
    
    Parameters:
    file_paths (list): List of file paths to process.

    Returns:
    None
    '''
    print(f'Running filter_stations on {file_paths}')
    start_time = time.time()

    dtype_stations = load_dtype('./config/dtype_stations.txt')
    stations_data = pd.read_csv('./config/stations.csv', dtype=dtype_stations)
    tx_set = create_set_for_tx(stations_data)

    with Pool(min(cpu_count(), 61)) as pool:
        pool.starmap(process_file, [(file_path, tx_set) for file_path in file_paths])

    elapsed_time = time.time() - start_time
    print(f"Took {elapsed_time//86400} days, {elapsed_time//3600%24} hrs, {elapsed_time//60%60} mins, {elapsed_time%60:.2f} secs")
