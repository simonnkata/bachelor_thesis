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
    print(f'Expected_distance: {expected_distance}')
    inverted = -recording
    min_indices, _ = find_peaks(inverted, distance=expected_distance - 3)

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
    systolic_peak = None
    dicrotic_notch = None
    dicrotic_peak = None
    for i in range(1,len(cycle)):
        if cycle[i] < cycle[i - 1]:
            systolic_peak = i-1
            break
    for i in range(systolic_peak + 1, len(cycle)):
        if cycle[i] > cycle[i - 1]:
            dicrotic_notch = i-1
            break
    if dicrotic_notch:
        for i in range(dicrotic_notch + 1, len(cycle)):
            if cycle[i] < cycle[i - 1]:
                dicrotic_peak = i-1
                break
    return [systolic_peak, dicrotic_notch, dicrotic_peak]


def fiducial_point_features(cycle):
    cycle_feature = {}
    on = cycle[0]
    sp_i, dn_i, dp_i = find_fiducial_points(cycle)
    sp = cycle[sp_i] if sp_i is not None else None
    dn = cycle[dn_i] if dn_i is not None else None
    dp = cycle[dp_i] if dp_i is not None else None
    pulse_interval = len(cycle) / fs
    cycle_feature['pulse_interval'] = pulse_interval
    if sp:
        # systolic width
        value_at_50 = on + 0.5 * (sp - on)
        segment = cycle[:sp_i + 1]
        index_at_50 = min(range(len(segment)), key=lambda i: abs(segment[i] - value_at_50))
        systolic_width = (sp_i - index_at_50) / fs

        # systolic_peak_time
        systolic_peak_time = sp_i / fs

        cycle_feature['systolic_width'] = systolic_width
        cycle_feature['systolic_peak_time'] = systolic_peak_time
    if dn:
        # diastolic width
        value_at_50 = on + 0.5 * (dn - sp)
        segment = cycle[sp_i:dn_i + 1]
        index_at_50 = min(range(len(segment)), key=lambda i: abs(segment[i] - value_at_50))
        diastolic_width = (dn_i - index_at_50) / fs

        # systolic time
        systolic_time = dn_i / fs

        # diastolic time
        diastolic_time = (len(cycle) - dn_i) / fs

        cycle_feature['diastolic_width'] = diastolic_width
        cycle_feature['systolic_time'] = systolic_time
        cycle_feature['diastolic_time'] = diastolic_time
    if dp:
        # time delay
        time_delay = (dp_i-sp_i) / fs

        # diastolic peak time
        diastolic_peak_time = dp_i / fs

        cycle_feature['time_delay'] = time_delay
        cycle_feature['diastolic_peak_time'] = diastolic_peak_time
    return cycle_feature


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
'''
data = load_pos_filter_plot()
df = split_data(data)
df = df.rename(columns={"class": "classification"})
for recording in df['signal']:
    # recording = df['signal'].iloc[0]
    cycles = split_cycles(recording)
    for cycle in cycles:
        points = find_fiducial_points(cycle)
        print(points)
        plt.clf()
        plt.plot(cycle)
        if points[0]:
            plt.scatter([points[0]], [cycle[points[0]]], color='red', label='sp')
            plt.legend()
        if points[1]:
            plt.scatter([points[1]], [cycle[points[1]]], color='green', label='dn')
            plt.legend()
        if points[2]:
            plt.scatter([points[2]], [cycle[points[2]]], color='blue', label='dp')
            plt.legend()
        plt.show()
'''