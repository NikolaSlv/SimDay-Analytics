import os
import time
from scripts import populate_na
from scripts import filter_stations
from scripts import gen_matrices
from scripts import find_closest

def generate():
    print('Running generate')
    start_time = time.time()

    file_paths_populate_na = []
    file_paths_filter_stations = []
    file_paths_gen_matrices = []
    for dir in os.listdir('./data'):
        for file in os.listdir(f'./data/{dir}'):
            if file.endswith('.csv'):
                # Generate path lists
                path = f'./data/{dir}/{file}'
                file_paths_populate_na.append(path)
                file_paths_filter_stations.append(path.replace('data', 'data_populated'))
                file_paths_gen_matrices.append(path.replace('data', 'data_filtered'))
    
    # Run using multiprocessing
    populate_na.run_parallel(file_paths_populate_na)
    filter_stations.run_parallel(file_paths_filter_stations)
    gen_matrices.run_parallel(file_paths_gen_matrices)
    
    elapsed_time = time.time() - start_time
    print('All data processed')
    print(f"Took {elapsed_time//86400} days, {elapsed_time//3600%24} hrs, {elapsed_time//60%60} mins, {elapsed_time%60:.2f} secs")

if __name__ == '__main__':
    q = input('Do you want to generate the matrices? (y/n): ')
    if q == 'y':
        generate()
    while True:
        target = input('Enter the target day (YYYY-MM-DD) or q to quit: ')
        if target == 'q':
            break
        which = input('Enter which set of features to compare (mat1/mat2/mat3): ')
        find_closest.run(target, which)
