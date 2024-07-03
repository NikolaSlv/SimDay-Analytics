import pandas as pd
import time
import os
from multiprocessing import Pool, cpu_count

CHUNK_SIZE = 1000000

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

def process_chunk(chunk, output_file, header):
    '''
    Process a single chunk of data by filling missing values and writing to output file.
    
    Parameters:
    chunk (DataFrame): The chunk of data to process.
    output_file (str): The path to the output file.
    header (bool): Whether to write the header to the output file.

    Returns:
    None
    '''
    chunk.ffill(inplace=True)
    chunk.to_csv(output_file, mode='a', index=False, header=header)

def process_file(input_file, dtype):
    '''
    Process a single file by reading in chunks, processing each chunk, and writing to output file.

    Parameters:
    input_file (str): The path to the input file.
    dtype (dict): The data types for the columns.

    Returns:
    None
    '''
    output_file = input_file.replace('data', 'data_populated')
    year = os.path.basename(input_file)[:4]
    if not os.path.exists(f'./data_populated/{year}'):
        os.makedirs(f'./data_populated/{year}')
    
    header = True
    for chunk in pd.read_csv(input_file, dtype=dtype, chunksize=CHUNK_SIZE):
        process_chunk(chunk, output_file, header)
        header = False

def run_parallel(file_paths):
    '''
    Run the processing in parallel for multiple files.
    
    Parameters:
    file_paths (list): List of file paths to process.

    Returns:
    None
    '''
    print(f'Running populate_na on {file_paths}')
    start_time = time.time()

    dtype = load_dtype('./config/dtype.txt')

    with Pool(min(cpu_count(), 61)) as pool:
        pool.starmap(process_file, [(file_path, dtype) for file_path in file_paths])

    elapsed_time = time.time() - start_time
    print(f"Took {elapsed_time//86400} days, {elapsed_time//3600%24} hrs, {elapsed_time//60%60} mins, {elapsed_time%60:.2f} secs")
