import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from filter import general_filter
from typing import List

mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g'}
classes = {0: 'baseline', 1: 'full', 2: 'moderate', 3: 'light'}

# Defines structure of custom Subjects Types
Subject_df = List[pd.DataFrame]
Subjects_df = List[Subject_df]

Subject_nd = List[np.ndarray]
Subjects_nd = List[Subject_nd]


def load_data() -> Subjects_df:
    """
        Iterative: Loads all the recorded data into a 2d list of dataframes.
    """
    _data = []
    files = sorted([f for f in os.listdir('data') if f.isdigit()], key=int)
    print(files)
    for subject in files:
        index = len(_data)
        _data.append([])
        print(f'Loading subject {subject}...')
        for filename in sorted(os.listdir('data/' + subject)):
            if filename.endswith('.csv'):
                file_path = os.path.join('data/' + subject, filename)
                _df = pd.read_csv(file_path)
                _data[index].append(_df)
    return _data


def plot_and_save_3d(df: pd.DataFrame, subject: str, variant: str, directory: str) -> None:
    """
        Utility: Plots and saves 3d recording
    """
    plt.clf()
    plt.plot(df['R'], color='red', label='R')
    plt.plot(df['G'], color='green', label='G')
    plt.plot(df['B'], color='blue', label='B')
    plt.title('Subject: ' + subject + ', Variant: ' + variant)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    save_dir = os.path.join(directory, variant + '.png')
    plt.savefig(save_dir)


def plot_and_save_1d(np_array: np.ndarray, subject: str, variant: str, directory: str) -> None:
    """
        Utility: Plots and saves 1d recording
    """
    plt.clf()
    plt.plot(np_array)
    plt.title('Subject: ' + subject + ', Variant: ' + variant)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    save_dir = os.path.join(directory, variant + '.png')
    plt.savefig(save_dir)


def plot_all(target_folder: str, _data: Subject_nd | Subjects_df, dimensions: int = 1):
    """
        Iterative: iterates through data object, plots and saves recordings in folders
    """
    os.makedirs(target_folder, exist_ok=True)
    for subject_number, subject_data in enumerate(_data):
        print(f"Plotting Subject {subject_number + 1}")
        directory = os.path.join(target_folder, str(subject_number + 1))
        os.makedirs(directory)
        if dimensions == 3:
            for index, recording in enumerate(subject_data):
                plot_and_save_3d(recording, str(subject_number + 1), mapping[index], directory)
        else:
            for index, recording in enumerate(subject_data):
                plot_and_save_1d(recording, str(subject_number + 1), mapping[index], directory)


def pos(_df: pd.DataFrame) -> np.ndarray:
    """
        Utility: Applies pos to 3d recordings in df object (R,G,B)
    """
    rgb = _df[['R', 'G', 'B']].values
    mean_rgb = np.mean(rgb, axis=0)
    normalized = (rgb - mean_rgb) / mean_rgb
    p = np.array([[0, 1, -1], [-2, 1, 1]])
    s = np.dot(normalized, p.T)
    alpha = np.std(s[:, 0]) / np.std(s[:, 1])
    signal = s[:, 0] - alpha * s[:, 1]
    return signal


def apply_pos(_data: Subjects_df) -> Subjects_nd:
    """
       Iterative: Iterates through df object, applies pos algorithm to each recording
    """
    all_recordings = []
    for subject in _data:
        subject_recordings = []
        for recording in subject:
            signal = pos(recording)
            subject_recordings.append(signal)
        all_recordings.append(subject_recordings)
    return all_recordings


def filter_data(_data: Subjects_nd) -> None:
    """
        Iterative: Applies a selected filter type to each recording
    """
    for subject in _data:
        for i in range(len(subject)):
            subject[i] = general_filter(subject[i], 30, 4, 5, 40, 0.5, 4, 0.5, 'band-pass')
    return


def get_min_max(recordings: Subjects_nd) -> List:
    """
        Iterative : Finds min and max points overall
    """
    global_min, global_max = np.inf, -np.inf
    for subject in recordings:
        for recording in subject:
            current_min = np.min(recording)
            current_max = np.max(recording)
            global_max = max(current_max, global_max)
            global_min = min(current_min, global_min)
    return [global_min, global_max]


def normalise(recordings: Subjects_nd) -> None:
    """
        Iterative: Applies min-max normalisation to all recordings
    """
    global_min, global_max = get_min_max(recordings)
    for subject_idx, subject in enumerate(recordings):
        for rec_idx, recording in enumerate(subject):
            recording = 2 * ((recording - global_min) / (global_max - global_min)) - 1
            recording = (recording - np.mean(recording)) / np.std(recording)
            recordings[subject_idx][rec_idx] = recording
