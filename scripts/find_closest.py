import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from sklearn.decomposition import PCA

def load_matrix(file_path):
    '''
    Load a matrix from a file.

    Parameters:
    file_path (str): The path to the file containing the matrix.

    Returns:
    numpy.ndarray: The matrix.
    '''
    return np.load(file_path)

def reduce_dimensionality(matrix):
    '''
    Reduce the dimensionality of a matrix using PCA.

    Parameters:
    matrix (numpy.ndarray): The matrix to reduce.

    Returns:
    numpy.ndarray: The reduced matrix.
    '''
    pca = PCA(n_components=min(matrix.shape[0], matrix.shape[1]))
    reduced_matrix = pca.fit_transform(matrix.T)
    return reduced_matrix.T

def compare_days_frobenius_norm(matrix1, matrix2):
    '''
    Compare two days using the Frobenius norm.

    Parameters:
    matrix1 (numpy.ndarray): The matrix for the first day.
    matrix2 (numpy.ndarray): The matrix for the second day.

    Returns:
    float: The Frobenius norm between the two matrices.
    '''
    return np.linalg.norm(matrix1 - matrix2, 'fro')

def plot_comparison(days, date1, date2):
    '''
    Plot the comparison between two days.

    Parameters:
    days (dict): A dictionary containing the matrices for each day.
    date1 (str): The first date.
    date2 (str): The second date.

    Returns:
    None
    '''
    matrix1 = days[date1].T
    matrix2 = days[date2].T

    features = matrix1.shape[0]
    columns = matrix1.shape[1]
    hours = 24
    columns_per_hour = columns // hours

    root = tk.Tk()
    root.title("Scrollable Plot")

    fig, axes = plt.subplots(features, 2, figsize=(15, features * 2), sharex=True)
    fig.suptitle(f'Comparison between {date1} and {date2}', fontsize=16)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update_plot(hour):
        start_col = hour * columns_per_hour
        end_col = min((hour + 1) * columns_per_hour, columns)
        for i in range(features):
            axes[i, 0].cla()
            axes[i, 1].cla()

            # Bar plot for Feature i from matrix1 and matrix2 with transparency
            axes[i, 0].bar(range(end_col - start_col), matrix1[i, start_col:end_col], label=f'{date1}', alpha=0.5, color='blue')
            axes[i, 0].bar(range(end_col - start_col), matrix2[i, start_col:end_col], label=f'{date2}', alpha=0.5, color='orange')
            axes[i, 0].set_title(f'Feature {i+1}')
            axes[i, 0].set_ylabel('Value')
            axes[i, 0].legend()

            # Bar plot the difference with transparency
            axes[i, 1].bar(range(end_col - start_col), matrix1[i, start_col:end_col] - matrix2[i, start_col:end_col], label='Difference', alpha=0.5, color='green')
            axes[i, 1].set_title(f'Feature {i+1} Difference')
            axes[i, 1].set_ylabel('Difference')
            axes[i, 1].legend()

        for ax in axes.flat:
            ax.set_xlabel('Column')

        canvas.draw()

    slider = tk.Scale(root, from_=0, to=hours-1, orient=tk.HORIZONTAL, length=600, label="Hour", command=lambda x: update_plot(int(x)))
    slider.pack(side=tk.BOTTOM)

    def on_closing():
        plt.close('all')
        root.destroy()
        root.quit()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)

    update_plot(0)
    root.mainloop()

def load_days(year, month, which):
    '''
    Load the days from the output path.

    Parameters:
    year_month (str): The year and month to load the days for.

    Returns:
    dict: A dictionary containing the days.
    '''
    days = {}
    for file in os.listdir(f'./output/{year}/{month}/'):
        if file.endswith('.npy') and file.split('_')[0] == which:
            print(f'Loading matrix from {file}')
            matrix = load_matrix(f'./output/{year}/{month}/{file}')
            days[file.split('_')[1].split('.')[0]] = matrix
    return days

def load_all():
    '''
    Load all the days from the output path.

    Returns:
    dict: A dictionary containing the days.
    '''
    days = {}
    for dir in os.listdir('./output'):
        for dir_month in os.listdir(f'./output/{dir}'):
            for file in os.listdir(f'./output/{dir}/{dir_month}'):
                if file.endswith('.npy'):
                    matrix = load_matrix(f'./output/{dir}/{dir_month}/{file}')
                    days[file.split('_')[1].split('.')[0]] = matrix
    return days

def find_closest_day(days, target_day):
    '''
    Find the closest day to the target day.

    Parameters:
    days (dict): A dictionary containing the days.
    target_day (str): The target day.

    Returns:
    tuple: The closest day and the distance to it.
    '''
    closest_day = None
    closest_distance = float('inf')
    for key in days.keys():
        if key == target_day:
            continue
        distance = compare_days_frobenius_norm(days[target_day], days[key])
        if distance < closest_distance:
            closest_distance = distance
            closest_day = key
    return closest_day, closest_distance

def run(target, which):
    print(f'Running find_closest for {target} comparing for {which}')
    start = time.time()

    days = load_all()
    
    print(f'Target day: {target}')
    print(pd.DataFrame(days[target]))
    
    closest_day, closest_distance = find_closest_day(days, target)

    print(f'Closest day to {target} is {closest_day} with distance {closest_distance}')
    print('Matrix for the closest day:')
    print(pd.DataFrame(days[closest_day]))
    
    plot_comparison(days, target, closest_day)

    time_elapsed = time.time() - start
    print(f'Took {time_elapsed//86400} days, {time_elapsed//3600%24} hrs, {time_elapsed//60%60} mins, {time_elapsed%60:.2f} secs')
