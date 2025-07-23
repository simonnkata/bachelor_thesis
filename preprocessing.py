import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from filter import general_filter
import statistics
import copy
import pickle

mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g'}
classes = {0: 'baseline', 1: 'full', 2: 'moderate', 3: 'light'}


# Load In the Data
# Loads all the recorded data into a 2d list of dataframes.
# Format: data[subject][a,b,c,d,e,f,g]
def load_data() -> list:
    _data = []
    files = sorted([f for f in os.listdir('data') if f.isdigit()], key=int)
    print(files)
    for subject in files:
        index = len(_data)
        _data.append([])
        print(f'Processing subject {subject}...')
        for filename in os.listdir('data/' + subject):
            if filename.endswith('.csv'):
                file_path = os.path.join('data/' + subject, filename)
                # print(f'Processing {file_path}...')
                _df = pd.read_csv(file_path)
                _data[index].append(_df)
    return _data


# Helper: plots and saves independent recording
def plot_and_save(np_array, subject, variant, directory) -> None:
    plt.clf()
    # plt.plot(df['R'], color='red', label='R')
    # plt.plot(df['G'], color='green', label='G')
    # plt.plot(df['B'], color='blue', label='B')
    plt.plot(np_array)
    plt.title('Subject: ' + subject + ', Variant: ' + variant)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    save_dir = os.path.join(directory, variant + '.png')
    plt.savefig(save_dir)


# Iterative: iterates through data object, plots and saves recordings in folders
def plot_all(folder, _data):
    os.makedirs(folder, exist_ok=True)
    for subject_number, subject_data in enumerate(_data):
        print(f"Plotting Subject {subject_number + 1}")
        directory = os.path.join(folder, str(subject_number + 1))
        os.makedirs(directory)
        for index, recording in enumerate(subject_data):
            plot_and_save(recording, str(subject_number + 1), mapping[index], directory)


# Helper
def snr(original, filtered):
    noise = original - filtered
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    _snr = 10 * np.log10(signal_power / noise_power)
    return _snr


# Iterative
def total_snr(original, filtered):
    snr_values = []
    for subject_o, subject_f in zip(original, filtered):
        for entry_o, entry_f in zip(subject_o, subject_f):
            snr_red = snr(entry_o['R'], entry_f['R'])
            snr_blue = snr(entry_o['B'], entry_f['B'])
            snr_green = snr(entry_o['G'], entry_f['G'])
            snr_values.extend([snr_red, snr_blue, snr_green])
            snr_values.append(snr_green)
    print(f"Mean: {np.mean(snr_values)}, Standard Deviation: {statistics.stdev(snr_values)}")
    return [np.mean(snr_values), statistics.stdev(snr_values)]


def filter_data(_data):
    for subject in _data:
        for i in range(len(subject)):
            subject[i] = general_filter(subject[i], 30, 4, 5, 40, 0.5, 4, 0.5, 'band-pass')
    return _data


# Helper: Applies pos to df object (R,G,B)
def pos(_df):
    rgb = _df[['R', 'G', 'B']].values
    mean_rgb = np.mean(rgb, axis=0)
    normalized = (rgb - mean_rgb) / mean_rgb
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    S = np.dot(normalized, P.T)
    alpha = np.std(S[:, 0]) / np.std(S[:, 1])
    signal = S[:, 0] - alpha * S[:, 1]
    return signal


# Iterative
def apply_pos(_data):
    all_recordings = []
    for subject in _data:
        subject_recordings = []
        for recording in subject:
            signal = pos(recording)
            subject_recordings.append(signal)
        all_recordings.append(subject_recordings)
    return all_recordings


# Helper
def get_min_max(recordings):
    global_min, global_max = np.inf, -np.inf
    for subject in recordings:
        for recording in subject:
            current_min = np.min(recording)
            current_max = np.max(recording)
            global_max = max(current_max, global_max)
            global_min = min(current_min, global_min)
    return [global_min, global_max]


# Iterative: Normalises Data throughout the object, on a scale covering the entire structure
def normalise(recordings):
    global_min, global_max = get_min_max(recordings)
    normalised_recordings = []
    for subject in recordings:
        subject_recording = []
        for recording in subject:
            recording = 2 * ((recording - global_min) / (global_max - global_min)) - 1
            subject_recording.append(recording)
        normalised_recordings.append(subject_recording)
    return normalised_recordings


# General: Loads data, Applies POS, Filters, Normalises
def load_pos_filter_plot():
    data = load_data()
    data_copy = copy.deepcopy(data)
    post_pos = apply_pos(data_copy)
    #    plot_all('corrected_version_pos', post_pos)
    filtered_data = filter_data(post_pos)
    #    plot_all('corrected_version_filter', post_pos)
    return filtered_data


# Loads POSed, filtered, and normalised data, splits into time blocks.
def split_data():
    with open('normalised.pkl', 'rb') as file:
        recordings = pickle.load(file)
    _df = pd.DataFrame(columns=['signal', 'class'])
    for subject in recordings[9:]:
        for index, recording in enumerate(subject[3:]):
            if index == 0:
                for i in range(0, 1800, 300):  # because recording 'd' lasts only 1 minute
                    new_row = pd.DataFrame({
                        'signal': [recording[i:i + 301]],
                        'class': index
                    })
                    _df = pd.concat([_df, new_row], ignore_index=True)
            else:
                for i in range(0, 3600, 300):
                    new_row = pd.DataFrame({
                        'signal': [recording[i:i + 301]],
                        'class': index
                    })
                    _df = pd.concat([_df, new_row], ignore_index=True)
    return _df


