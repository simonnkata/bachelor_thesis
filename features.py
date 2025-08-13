# Features:
# Pulse Rate Variability
# Amplitude of Peaks
# Rise Time
# Decay Time

import pickle
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from validation import heart_rate
Cfrom preprocessing import load_pos_filter_plot, split_data
import time
# Fudicial points: Onset, Systolic Peak, Max Slope, Dicrotic Notch, Diastolic Peak
fs = 30
classes = {0: 'baseline', 1: 'full', 2: 'moderate', 3: 'light'}

def find_pulse_rate_variability(entry):
    recording = entry['signal']
    expected_distance = int(fs * 60 / heart_rate(recording, fs))
    peaks, _ = find_peaks(recording, distance=expected_distance)
    ibi = np.diff(peaks) / fs
    prv_std = np.std(ibi)
    return prv_std


def find_peak_amplitude(entry):
    recording = entry['signal']
    expected_distance = int(fs * 60 / heart_rate(recording, fs))
    peaks, _ = find_peaks(recording, distance=expected_distance)
    peak_amplitudes = recording[peaks]
    return np.std(peak_amplitudes)


def find_rise_time(entry):
    return 0


def find_decay_time(entry):
    return 0

def extract():
    data = load_pos_filter_plot()
    df = split_data(data)
    df = df.rename(columns={"class": "classification"})
    # with open('windowed_df.pkl', 'rb') as f:
    #    df = pickle.load(f)
    #    df = df.rename(columns={"class": "classification"})

    peak_amplitude = df.apply(find_peak_amplitude, axis=1)
    rise_time = df.apply(find_rise_time, axis=1)
    decay_time = df.apply(find_decay_time, axis=1)
    pulse_rate = df.apply(find_pulse_rate_variability, axis=1)
    features_df = pd.DataFrame({
        'peak_amplitude': peak_amplitude,
        'rise_time': rise_time,
        'decay_time': decay_time,
        'pulse_rate': pulse_rate,
        'classification': df.classification
    })
    print(f'We are working with {len(features_df)} rows')
    return features_df
start = time.time()
extract()
end = time.time()
print(f"Features Extracted in {end - start} seconds")