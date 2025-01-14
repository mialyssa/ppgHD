from importlib import resources as impresources
from . import demo_data
import pickle
import math
import numpy as np


# PPGDalia_S6_stairs
demo_data_file_train = impresources.files(demo_data) / 'S5slimmed_dalia_aligned_prefiltered_80000.pkl'
demo_data_filename_train = demo_data_file_train.resolve()

demo_data_file_test = impresources.files(demo_data) / 'S5slimmed_dalia_aligned_prefiltered_80000.pkl'
demo_data_filename_test = demo_data_file_test.resolve()


def load_demo_data(activity=None, split='train'):
    """
    Loads demo data, filters by specific activities, and splits into train/test sets.

    Args:
        activity (int or list of int, optional): Activity label(s) to filter data. If None, returns all data.
        split (str): 'train' or 'test' to load respective split.

    Returns:
        X (numpy.ndarray): Filtered signal data.
        y (numpy.ndarray): Filtered labels (e.g., heart rate).
        act (numpy.ndarray): Activity labels corresponding to the data.
    """
    # Choose file based on split
    demo_data_filename = demo_data_filename_train if split == 'train' else demo_data_filename_test

    # Load data
    with open(demo_data_filename, 'rb') as handle:
        data = pickle.load(handle)

    X, y, act = data['X'], data['y'], data.get('act', None)  # 'act' may not exist in all datasets

    # If activity is specified (single or multiple), filter the entire dataset by activity first
    if activity is not None and act is not None:
        if isinstance(activity, list):  # Check if activity is a list of activities
            activity_indices = np.isin(act, activity)  # Get indices for multiple activities
        else:
            activity_indices = (act == activity)  # Get indices for a single activity
        X, y, act = X[activity_indices], y[activity_indices], act[activity_indices]

    # Update lenX after filtering
    lenX = len(X)

    # Determine split indices
    split_idx = math.floor(lenX * 0.8)
    # Print the total number of data points if split == 'train'
    if split == 'train':
        print(f"Total number of data points: {lenX}")
        print(f"Total number of train data: {split_idx}")
        X, y, act = X[:split_idx], y[:split_idx], act[:split_idx] if act is not None else None
        print("Finished train data splitting")
    else:
        print(f"Total number of test data: {lenX - split_idx}")
        X, y, act = X[split_idx:], y[split_idx:], act[split_idx:] if act is not None else None
        print("Finished test data splitting")

    return X, y, act
