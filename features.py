# Features:
# Pulse Rate Variability
# Amplitude of Peaks
# Rise Time
# Decay Time

import pickle
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from validation import heart_rate, validate
import time
import matplotlib.pyplot as plt
import uuid

# Fudicial points: Onset, Systolic Peak, Max Slope, Dicrotic Notch, Diastolic Peak
fs = 30
classes = {0: 'baseline', 1: 'full', 2: 'moderate', 3: 'light'}
features_list = ['pulse_interval', 'systolic_width', 'systolic_peak_time', 'diastolic_width',
                 'systolic_time', 'diastolic_time', 'time_delay', 'diastolic_peak_time']


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
    alternatives = find_cycles(inverted, distance=expected_distance)
    '''
    plt.clf()
    plt.plot(recording)
    for onset in onsets:
        plt.axvline(x=onset, color='red', linestyle='--', linewidth=1.5)
    for alternative in alternatives:
        plt.axvline(x=alternative, color='blue', linestyle='--', linewidth=1.5)
    plt.title("rPPG Signal with Detected Cycles")
    plt.savefig(f"huh/{uuid.uuid4().hex}.png")
    '''
    result = []
    for i in range(1, len(onsets)):
        cycle = recording[onsets[i - 1]:onsets[i]]
        result.append(cycle)
    return result


def find_cycles(inverted, distance):
    onsets = []
    for i in range(distance, len(inverted), distance):
        # check to the left for the next biggest value
        # check to the right for the next biggest value
        # take the shorter one, or take the value at i if shorter one is beyond our tolerance
        tolerance = 5
        left, right = 0, 0
        while i - left >= 0 and inverted[i - left] <= inverted[i] and left < distance:
            left += 1
        while i + right < len(inverted) and inverted[i + right] <= inverted[i] and right < distance:
            right += 1
        if min(left, right) > tolerance:
            point = i
        elif right < left:
            point = i + right
        else:
            point = i - left
        onsets.append(point)
    return onsets


def find_fiducial_points(cycle):
    systolic_peak = None
    dicrotic_notch = None
    dicrotic_peak = None
    for i in range(1, len(cycle)):
        if cycle[i] < cycle[i - 1]:
            systolic_peak = i - 1
            break
    for i in range(systolic_peak + 1, len(cycle)):
        if cycle[i] > cycle[i - 1]:
            dicrotic_notch = i - 1
            break
    if dicrotic_notch:
        for i in range(dicrotic_notch + 1, len(cycle)):
            if cycle[i] < cycle[i - 1]:
                dicrotic_peak = i - 1
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

        cycle_feature['systolic_width'] = abs(systolic_width)
        cycle_feature['systolic_peak_time'] = abs(systolic_peak_time)
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

        cycle_feature['diastolic_width'] = abs(diastolic_width)
        cycle_feature['systolic_time'] = abs(systolic_time)
        cycle_feature['diastolic_time'] = abs(diastolic_time)
    if dp:
        # time delay
        time_delay = (dp_i - sp_i) / fs

        # diastolic peak time
        diastolic_peak_time = dp_i / fs

        cycle_feature['time_delay'] = abs(time_delay)
        cycle_feature['diastolic_peak_time'] = abs(diastolic_peak_time)
    return cycle_feature


def aggregate_fiducial_features(recording):
    cycles = split_cycles(recording)
    cycle_features = []
    for cycle in cycles:
        cycle_feature = fiducial_point_features(cycle)
        cycle_features.append(cycle_feature)
    features = ['pulse_interval', 'systolic_width', 'systolic_peak_time', 'diastolic_width',
                'systolic_time', 'diastolic_time', 'time_delay', 'diastolic_peak_time']
    features_from_cycles = dict.fromkeys(features)
    feature_rep = dict.fromkeys(features)
    trim_fraction = 0.1
    for feature in features:
        features_from_cycles[feature] = [x.get(feature) for x in cycle_features if x.get(feature) is not None]
        if features_from_cycles[feature]:
            length = len(features_from_cycles[feature])
            trim_size = int(length * trim_fraction)
            trimmed = sorted(features_from_cycles[feature])[trim_size: length - trim_size]
            feature_rep[feature] = np.mean(trimmed)
        else:
            feature_rep[feature] = None
    return feature_rep


def extract(df):
    # data = load_pos_filter_plot()
    # df = split_data(data)
    # df = df.rename(columns={"class": "classification"})
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
    for idx, row in df.iterrows():
        fiducial_features = aggregate_fiducial_features(row['signal'])
        features_df.at[idx, 'pulse_interval'] = fiducial_features['pulse_interval']
        features_df.at[idx, 'systolic_width'] = fiducial_features['systolic_width']
        features_df.at[idx, 'systolic_peak_time'] = fiducial_features['systolic_peak_time']
        features_df.at[idx, 'diastolic_width'] = fiducial_features['diastolic_width']
        features_df.at[idx, 'systolic_time'] = fiducial_features['systolic_time']
        features_df.at[idx, 'diastolic_time'] = fiducial_features['diastolic_time']
        features_df.at[idx, 'time_delay'] = fiducial_features['time_delay']
        features_df.at[idx, 'diastolic_peak_time'] = fiducial_features['diastolic_peak_time']

    print(f'We are working with {len(features_df)} rows')
    return features_df


'''
df = validate()
start = time.time()
extract(df)
end = time.time()

print(f"Features Extracted in {end - start} seconds")

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
'''
data = load_pos_filter_plot()
df = split_data(data)
df = df.rename(columns={"class": "classification"})
complete = 0
incomplete = 0
for recording in df['signal']:
    features = aggregate_fiducial_features(recording)
    has_features = all(feature in features and features[feature] is not None for feature in features_list)
    if has_features:
        complete += 1
    else:
        incomplete += 1
print(f'Complete Features: {complete}')
print(f'Incomplete Features: {incomplete}')
'''
