import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from filter import cheby2_filter, bessel_filter, butter_filter, cheby1_filter, ellip_filter, fir_filter
import statistics
import copy

mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g'}


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
                #print(f'Processing {file_path}...')
                df = pd.read_csv(file_path)
                _data[index].append(df)
    return _data


def plot_and_save(df, subject, variant, directory) -> None:
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


def plot_all(folder, _data):
    os.makedirs(folder, exist_ok=True)
    for subject_number, subject_data in enumerate(_data):
        print(f"Plotting Subject {subject_number + 1}")
        directory = os.path.join(folder, str(subject_number + 1))
        os.makedirs(directory)
        for index, recording in enumerate(subject_data):
            plot_and_save(recording, str(subject_number + 1), mapping[index], directory)


def snr(original, filtered):
    noise = original - filtered
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    _snr = 10 * np.log10(signal_power / noise_power)
    return _snr


def total_snr(original, filtered):
    snr_values = []
    for subject_o, subject_f in zip(original, filtered):
        for entry_o, entry_f in zip(subject_o, subject_f):
            snr_red = snr(entry_o['R'], entry_f['R'])
            snr_blue = snr(entry_o['B'], entry_f['B'])
            snr_green = snr(entry_o['G'], entry_f['G'])
            snr_values.extend([snr_red, snr_blue, snr_green])
    print(f"Mean: {np.mean(snr_values)}, Standard Deviation: {statistics.stdev(snr_values)}")
    return [np.mean(snr_values), statistics.stdev(snr_values)]


def filter_data(_data):
    for subject in _data:
        for i in range(len(subject)):
            subject[i] = cheby2_filter(30, 3, 4, 40, subject[i])
    return _data


data = load_data()
data_copy = copy.deepcopy(data)
filtered_data = filter_data(data_copy)
total_snr(data, filtered_data)
# plot_all('figures_v2', data)
