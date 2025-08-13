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
from preprocessing import load_pos_filter_plot, split_data
import time
import matplotlib.pyplot as plt

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


def split_cycles(recording):
    expected_distance = int(fs * 60 / heart_rate(recording, fs))
    peaks, _ = find_peaks(recording, distance=expected_distance)
    inverted = -recording
    min_indices, _ = find_peaks(inverted, distance=expected_distance)

    # Each min_index is an onset
    onsets = min_indices
    '''
    plt.plot(recording)
    plt.plot(peaks, recording[peaks], 'rx')
    plt.plot(onsets, recording[onsets], 'go')
    plt.title("rPPG Signal with Detected Cycles")
    plt.show()
    '''
    result = []
    for i in range(1, len(onsets)):
        cycle = recording[onsets[i - 1]:onsets[i]]
        result.append(cycle)
    return result


def find_fiducial_points(cycle):
    systolic_peak = np.argmax(cycle)
    dicrotic_notch = None
    dicrotic_peak = None
    for i in range(systolic_peak + 1, len(cycle)):
        if cycle[i] > cycle[i - 1]:
            dicrotic_notch = i
            break
    if dicrotic_notch:
        for i in range(dicrotic_notch + 1, len(cycle)):
            if cycle[i] < cycle[i - 1]:
                dicrotic_peak = i
                break
    return [systolic_peak, dicrotic_notch, dicrotic_peak]


def fiducial_point_features(recording):
    cycles = split_cycles(recording)
    cycle_features = []
    for cycle in cycles:
        cycle_feature = {}
        on = cycle[0]
        sp_i, dn_i, dp_i = find_fiducial_points(cycle)
        sp, dn, dp = cycle[sp_i], cycle[dn_i], cycle[dp_i]
        pulse_interval = len(cycle) / fs
        cycle_feature['pulse_interval'] = pulse_interval
        if sp:
            # systolic with
            value_at_50 = on + 0.5 * (sp - on)
            segment = cycle[:sp_i+1]
            index_at_50 = min(range(len(segment)), key=lambda i: abs(segment[i] - value_at_50))
            systolic_width = (sp_i-index_at_50)/fs

            # systolic_peak_time
            systolic_peak_time = sp_i/fs

            cycle_feature['systolic_width'] = systolic_width
            cycle_feature['systolic_peak_time'] = systolic_peak_time
        if dn:
            diastolic_width = 0
            systolic_time = 0
            diastolic_time = 0
            cycle_feature['diastolic_width'] = diastolic_width
            cycle_feature['systolic_time'] = systolic_time
            cycle_feature['diastolic_time'] = diastolic_time
        if dp:
            time_delay = 0
            diastolic_peak_time = 0
            cycle_feature['time_delay'] = time_delay
            cycle_feature['diastolic_peak_time'] = diastolic_peak_time
    # Do analysis with cycle_features


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
