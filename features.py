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

fs = 30
classes = {0: 'baseline', 1: 'full', 2: 'moderate', 3: 'light'}

# Load windowed_data
with open('windowed_df.pkl', 'rb') as f:
    df = pickle.load(f)

feature_df = pd.DataFrame(columns=['features', 'class'])
for index, row in df.iterrows():
    print(row['class'])


def pulse_rate_variability(recording):
    expected_distance = int(fs * 60 / heart_rate(recording, fs))
    peaks, _ = find_peaks(recording, distance=expected_distance)
    ibi = np.diff(peaks) / fs
    prv_std = np.std(ibi)
    return prv_std


def peak_amplitude(recording):
    expected_distance = int(fs * 60 / heart_rate(recording, fs))
    peaks, _ = find_peaks(recording, distance=expected_distance)
    peak_amplitudes = recording[peaks]
    return np.std(peak_amplitudes)


def rise_time():
    return 0


def decay_time():
    return 0

