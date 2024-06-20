import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from sklearn.decomposition import PCA
import os

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

    fig, axes = plt.subplots(features, 2, figsize=(15, features * 2), sharex=True)
    fig.suptitle(f'Comparison between {date1} and {date2}', fontsize=16)

    for i in range(features):
        axes[i, 0].plot(range(columns), matrix1[i], label=f'{date1}')
        axes[i, 0].plot(range(columns), matrix2[i], label=f'{date2}')
        axes[i, 0].set_title(f'Feature {i+1}')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].legend()

        axes[i, 1].plot(range(columns), matrix1[i] - matrix2[i], label='Difference')
        axes[i, 1].set_title(f'Feature {i+1} Difference')
        axes[i, 1].set_ylabel('Difference')
        axes[i, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel('Column')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.5)

    mpld3.show()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    cax1 = axes[0].imshow(matrix1, aspect='auto', cmap='viridis', interpolation='none')
    fig.colorbar(cax1, ax=axes[0])
    axes[0].set_title(f'Heatmap for {date1}')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Feature')

    cax2 = axes[1].imshow(matrix2, aspect='auto', cmap='viridis', interpolation='none')
    fig.colorbar(cax2, ax=axes[1])
    axes[1].set_title(f'Heatmap for {date2}')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Feature')

    plt.tight_layout()
    plt.show()

def load_days(year_month, which):
    '''
    Load the days from the output path.

    Parameters:
    year_month (str): The year and month to load the days for.

    Returns:
    dict: A dictionary containing the days.
    '''
    days = {}
    for file in os.listdir(f'./output/{year_month}/'):
        if file.endswith('.npy') and file.split('_')[0] == which:
            print(f'Loading matrix from {file}')
            matrix = load_matrix(f'./output/{year_month}/{file}')
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

def main():
    # Compare across matrix 3, wind conditions
    days = load_days('194001', 'mat3')

    # Reduce the dimensionality of the matrices
    # Might lead to loss of information
    # for key in days.keys():
    #     days[key] = reduce_dimensionality(days[key])

    target_day = '1940-01-25'
    print(f'Target day: {target_day}')
    print(pd.DataFrame(days[target_day]))
    
    closest_day, closest_distance = find_closest_day(days, target_day)

    print(f'Closest day to {target_day} is {closest_day} with distance {closest_distance}')
    print('Matrix for the closest day:')
    print(pd.DataFrame(days[closest_day]))
    
    plot_comparison(days, target_day, closest_day)

if __name__ == "__main__":
    main()
