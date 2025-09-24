# Features:
# Pulse Rate Variability
# Amplitude of Peaks
# Rise Time
# Decay Time

import pickle
import pandas as pd
import numpy as np
from scipy.ndimage import label
from scipy.signal import find_peaks, butter, filtfilt
from validation import heart_rate, validate
import statistics
import time
import matplotlib.pyplot as plt
import uuid
import seaborn as sns
import torch
import torch.nn as nn
import os

# Fudicial points: Onset, Systolic Peak, Max Slope, Dicrotic Notch, Diastolic Peak
fs = 30
classes = {0: 'baseline', 1: 'full', 2: 'moderate', 3: 'light'}
features_list = ['pulse_interval', 'systolic_width', 'systolic_peak_time', 'diastolic_width',
                 'systolic_time', 'diastolic_time', 'time_delay', 'diastolic_peak_time']


def apply_mask_and_balance(df, mask_type, balance=0):
    if mask_type == '2-class':
        df['classification'] = df['classification'].replace({
            'moderate': 'stenosis',
            'light': 'stenosis',
            'full': 'stenosis'
        })
    elif mask_type == '3-class':
        df = df[df['classification'] != 'baseline']
    elif mask_type == '3-class-b':
        df['classification'] = df['classification'].replace({
            'moderate': 'medium',
            'light': 'medium',
            'full': 'full'
        })
    if balance:
        grouped = df.groupby('classification')
        min_size = grouped.size().min()
        df = grouped.apply(lambda x: x.sample(n=min_size)).reset_index(drop=True)
    return df


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


def find_mean(entry):
    recording = entry['signal']
    return statistics.mean(recording)


def find_standard_deviation(entry):
    recording = entry['signal']
    return statistics.stdev(recording)


def find_area_under_curve(entry):
    recording = entry['signal']
    return sum(abs(x) for x in recording) / len(recording)


def find_root_mean_square(entry):
    recording = entry['signal']
    return np.sqrt(np.mean(recording ** 2))


def find_shape_factor(entry):
    recording = entry['signal']
    rms = find_root_mean_square(entry)
    mean_abs = np.mean(np.abs(recording))
    return rms / mean_abs


def find_min_max_deviation(entry):
    recording = entry['signal']
    return abs(max(recording) - min(recording))


def split_cycles(reference_recording, cycled_recording):
    expected_distance = int(fs * 60 / heart_rate(reference_recording, fs))
    inverted = -reference_recording
    min_indices, _ = find_peaks(inverted, distance=expected_distance - 3)

    # Each min_index is an onset
    onsets = min_indices
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
        cycle = cycled_recording[onsets[i - 1]:onsets[i]]
        result.append(cycle)
    return result


def _bandpass(x, low, high, fs, order=3):
    ny = 0.5 * fs
    b, a = butter(order, [low / ny, high / ny], btype='band')
    return filtfilt(b, a, x)


def split_cycles_improved(reference_recording, cycled_recording, fs):
    # safe HR -> expected distance in samples
    hr = heart_rate(reference_recording, fs)
    if hr <= 0:
        raise ValueError("heart_rate returned <= 0")
    expected = int(fs * 60.0 / hr)
    expected = max(3, expected)

    # preprocess: bandpass + normalize
    sig = _bandpass(reference_recording, 0.7, 3.5, fs)  # adjust band if needed
    sig = (sig - np.median(sig)) / (np.std(sig) + 1e-8)
    inv = -sig

    # adaptive prominence scale
    scale = np.ptp(inv)
    prominence = max(scale * 0.06, np.std(inv) * 0.4, 0.05)

    # initial loose detection (allow slightly smaller distance to avoid misses)
    peaks, props = find_peaks(inv, distance=max(1, int(expected * 0.55)), prominence=prominence)

    # if too few peaks, relax prominence once
    if len(peaks) < 2:
        peaks, props = find_peaks(inv, distance=max(1, int(expected * 0.45)), prominence=prominence * 0.35)

    # remove spuriously-close peaks: keep the one with larger prominence
    if len(peaks) > 1:
        prominences = props.get('prominences', np.ones_like(peaks))
        keep = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - keep[-1] < max(1, int(0.4 * expected)):  # too close
                # drop the smaller-prominence one
                if prominences[i] > prominences[np.where(peaks == keep[-1])[0][0]]:
                    keep[-1] = peaks[i]
            else:
                keep.append(peaks[i])
        peaks = np.array(keep)

    # fill large gaps by searching for missed peaks inside long intervals
    if len(peaks) > 1:
        diffs = np.diff(peaks)
        median = np.median(diffs)
        long_idx = np.where(diffs > 1.6 * median)[0]
        for idx in long_idx[::-1]:  # reverse so inserts don't shift later indices
            a, b = peaks[idx], peaks[idx + 1]
            segment = inv[a:b]
            if len(segment) < 4:
                continue
            sub_peaks, _ = find_peaks(segment, prominence=prominence * 0.25, distance=max(1, int(expected * 0.25)))
            if len(sub_peaks):
                # insert first reasonable subpeak (you can adjust to insert multiple)
                peaks = np.insert(peaks, idx + 1, a + sub_peaks[0])

    peaks = np.sort(peaks)

    # build cycles
    result = []
    for i in range(1, len(peaks)):
        start, end = peaks[i - 1], peaks[i]
        cycle = cycled_recording[start:end]
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


def find_fiducial_points(cycle, order):
    if order == 0:
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
    elif order == 1:
        # u: max_peak_in_systole, v: min_peak_after_max_in_systole, w: max_peak_in_diastole
        u = None
        v = None
        w = None
        for i in range(1, len(cycle)):
            if cycle[i] < cycle[i - 1]:
                u = i - 1
                break
        for i in range(u + 1, len(cycle)):
            if cycle[i] > cycle[i - 1]:
                v = i - 1
                break
        if v:
            for i in range(v + 1, len(cycle)):
                if cycle[i] < cycle[i - 1]:
                    w = i - 1
                    break
        return [u, v, w]
    elif order == 2:
        print(2)
    elif order == 3:
        print(3)


def fiducial_point_features(cycle, order):
    cycle_feature = {}
    on = cycle[0]
    if order == 0:
        sp_i, dn_i, dp_i = find_fiducial_points(cycle, order)
        sp = cycle[sp_i] if sp_i is not None else None
        dn = cycle[dn_i] if dn_i is not None else None
        dp = cycle[dp_i] if dp_i is not None else None
        pulse_interval = len(cycle) / fs
        cycle_feature['pulse_interval'] = pulse_interval
        '''
        plt.clf()
        plt.plot(cycle, label='cycle', linewidth=1.5)
        if sp:
            plt.scatter(sp_i, sp, color="red", zorder=5, label='Systolic Peak')
        if dn:
            plt.scatter(dn_i, dn, color="blue", zorder=5, label='Dicrotic Notch')
        if dp:
            plt.scatter(dp_i, dp, color="green", zorder=5, label='Diastolic Peak')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title('rPPG Cycle with Fiducial Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        files = os.listdir('plots/marked_plots/zero')
        numbers = [int(os.path.splitext(f)[0]) for f in files]
        image_number = max(numbers, default=0) + 1
        plt.savefig(f'plots/marked_plots/zero/{str(image_number)}.png')
        '''
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
    elif order == 1:
        u_i, v_i, w_i = find_fiducial_points(cycle, order)
        u = cycle[u_i] if u_i is not None else None
        v = cycle[v_i] if v_i is not None else None
        w = cycle[w_i] if w_i is not None else None
        '''
        plt.clf()
        plt.plot(cycle, label='cycle', linewidth=1.5)
        if u:
            plt.scatter(u_i, u, color="red", zorder=5, label="U point")
        if v:
            plt.scatter(v_i, v, color="blue", zorder=5, label="V point")
        if w:
            plt.scatter(w_i, w, color="green", zorder=5, label="W point")
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title('First Derivative of rPPG Cycle with Fiducial Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        files = os.listdir('plots/marked_plots/first')
        numbers = [int(os.path.splitext(f)[0]) for f in files]
        image_number = max(numbers, default=0) + 1
        plt.savefig(f'plots/marked_plots/first/{str(image_number)}.png')
        '''
        if u and v and w:
            cycle_feature['u_amplitude'] = u
            cycle_feature['v_amplitude'] = v
            cycle_feature['w_amplitude'] = w
            cycle_feature['v_w_amplitude_ratio'] = abs(w / v)
            cycle_feature['u_v_amplitude_ratio'] = abs(u / v)
            cycle_feature['u_w_amplitude_ratio'] = abs(u / w)

            cycle_feature['t_u_v'] = abs(v_i - u_i)
            cycle_feature['t_v_w'] = abs(w_i - v_i)
            cycle_feature['t_u_w'] = abs(w_i - u_i)

            cycle_feature['slope_u_v'] = abs((v - u) / (v_i - u_i))
            cycle_feature['slope_v_w'] = abs((w - v) / (w_i - v_i))
            cycle_feature['slope_u_w'] = abs((w - u) / (w_i - u_i))

            cycle_feature['t_u_v_ratio'] = abs((v_i - u_i) / (w_i - v_i))
            cycle_feature['t_v_w_ratio'] = abs((w_i - v_i) / (v_i - u_i))
    return cycle_feature


def aggregate_fiducial_features(order, cycles):
    cycle_features = []
    for cycle in cycles:
        cycle_feature = fiducial_point_features(cycle, order)
        cycle_features.append(cycle_feature)
    if order == 0:
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
    elif order == 1:
        features = ['u_amplitude', 'v_amplitude', 'w_amplitude', 'v_w_amplitude_ratio', 'u_v_amplitude_ratio',
                    'u_w_amplitude_ratio', 't_u_v', 't_v_w', 't_u_w', 'slope_u_v', 'slope_v_w', 'slope_u_w',
                    't_u_v_ratio', 't_v_w_ratio']
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
    else:
        return []


def extract(df):
    # data = load_pos_filter_plot()
    # df = split_data(data)
    # df = df.rename(columns={"class": "classification"})
    # with open('windowed_df.pkl', 'rb') as f:
    #    df = pickle.load(f)
    #    df = df.rename(columns={"class": "classification"})

    peak_amplitude = df.apply(find_peak_amplitude, axis=1)
    mean = df.apply(find_mean, axis=1)
    standard_deviation = df.apply(find_standard_deviation, axis=1)
    pulse_rate = df.apply(find_pulse_rate_variability, axis=1)
    area_under_curve = df.apply(find_area_under_curve, axis=1)
    root_mean_square = df.apply(find_root_mean_square, axis=1)
    shape_factor = df.apply(find_shape_factor, axis=1)
    min_max_deviation = df.apply(find_min_max_deviation, axis=1)

    features_df = pd.DataFrame({
        'peak_amplitude': peak_amplitude,
        'mean': mean,
        'standard_deviation': standard_deviation,
        'area_under_curve': area_under_curve,
        'pulse_rate': pulse_rate,
        # 'root_mean_square': root_mean_square,
        # 'shape_factor': shape_factor,
        # 'min_max_deviation': min_max_deviation,
        'classification': df.classification,
        'patient_id': df.patient_id,
    })
    for idx, row in df.iterrows():
        recording = row['signal']
        first_derivative = np.gradient(recording)
        # second_derivative = np.gradient(first_derivative)
        # third_derivative = np.gradient(second_derivative)
        cycles = split_cycles_improved(recording, recording, 30)
        # cycles = split_cycles(recording, recording)
        cycles_1 = split_cycles(recording, first_derivative)
        # cycles_2 = split_cycles(recording, second_derivative)
        # cycles_3 = split_cycles(recording, third_derivative)
        fiducial_features = aggregate_fiducial_features(0, cycles)
        fiducial_features_1 = aggregate_fiducial_features(1, cycles_1)
        # fiducial_features_2 = aggregate_fiducial_features(second_derivative, 2, cycles)
        # fiducial_features_3 = aggregate_fiducial_features(third_derivative, 3, cycles)

        features_df.at[idx, 'pulse_interval'] = fiducial_features['pulse_interval']
        features_df.at[idx, 'systolic_width'] = fiducial_features['systolic_width']
        features_df.at[idx, 'systolic_peak_time'] = fiducial_features['systolic_peak_time']
        features_df.at[idx, 'diastolic_width'] = fiducial_features['diastolic_width']
        features_df.at[idx, 'systolic_time'] = fiducial_features['systolic_time']
        features_df.at[idx, 'diastolic_time'] = fiducial_features['diastolic_time']
        features_df.at[idx, 'time_delay'] = fiducial_features['time_delay']
        features_df.at[idx, 'diastolic_peak_time'] = fiducial_features['diastolic_peak_time']


        features_df.at[idx, 'u_amplitude'] = fiducial_features_1['u_amplitude']
        features_df.at[idx, 'v_amplitude'] = fiducial_features_1['v_amplitude']
        features_df.at[idx, 'w_amplitude'] = fiducial_features_1['w_amplitude']

        features_df.at[idx, 'v_w_amplitude_ratio'] = fiducial_features_1['v_w_amplitude_ratio']
        features_df.at[idx, 'u_v_amplitude_ratio'] = fiducial_features_1['u_v_amplitude_ratio']
        features_df.at[idx, 'u_w_amplitude_ratio'] = fiducial_features_1['u_w_amplitude_ratio']

        features_df.at[idx, 't_u_v'] = fiducial_features_1['t_u_v']
        features_df.at[idx, 't_v_w'] = fiducial_features_1['t_v_w']
        features_df.at[idx, 't_u_w'] = fiducial_features_1['t_u_w']

        features_df.at[idx, 'slope_u_v'] = fiducial_features_1['slope_u_v']
        features_df.at[idx, 'slope_v_w'] = fiducial_features_1['slope_v_w']
        features_df.at[idx, 'slope_u_w'] = fiducial_features_1['slope_u_w']

        #features_df.at[idx, 't_u_v_ratio'] = fiducial_features_1['t_u_v_ratio']
        #features_df.at[idx, 't_v_w_ratio'] = fiducial_features_1['t_v_w_ratio']

    #features_df = apply_mask_and_balance(features_df, '2-class', 1)
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


def feature_embedding(df):
    def embed(signal):
        signal = torch.tensor(signal, dtype=torch.float32)
        signal_length = len(signal)
        patch_size = int(signal_length / 8)
        embedding_dim = 64
        num_heads = 4
        num_layers = 2

        tokens = signal.unfold(0, patch_size, patch_size)
        tokens = tokens.unsqueeze(0)
        linear_proj = nn.Linear(patch_size, embedding_dim)
        x = linear_proj(tokens)
        pos_encoding = nn.Parameter(torch.randn(1, x.size(1), embedding_dim))
        x = x + pos_encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        x = transformer_encoder(x)
        embedding = x.mean(dim=1)
        return embedding[0]

    embeddings = df['signal'].apply(lambda s: pd.Series(embed(s).detach().numpy()))
    embeddings.columns = [f'emb_{i}' for i in range(64)]
    result = pd.concat([embeddings, df[['classification', 'patient_id']]], axis=1)
    return result
