import pandas as pd
import time
import os

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

def run(input):
    print(f'Running populate_na on {input}')
    start_time = time.time()

    dtype = load_dtype('./config/dtype.txt')
    try:
        input_file = input
        output_file = input.replace('data', 'data_populated')

        year = os.path.basename(input)[:4]
        if not os.path.exists(f'./data_populated/{year}'):
            os.makedirs(f'./data_populated/{year}')
        
        header = True
        for chunk in pd.read_csv(input_file, dtype=dtype, chunksize=CHUNK_SIZE):
            # Fill missing values with the value of each previous row
            chunk.ffill(inplace=True)
            # Write the chunk to a new file
            chunk.to_csv(output_file, mode='a', index=False, header=header)
            header = False
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Took {elapsed_time//86400} days, {elapsed_time//3600%24} hrs, {elapsed_time//60%60} mins, {elapsed_time%60:.2f} secs")
