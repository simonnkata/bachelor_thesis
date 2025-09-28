import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from Validator.validator_helpers import heart_rate
import statistics

fs = 30
classes = {0: 'baseline', 1: 'full', 2: 'moderate', 3: 'light'}
features_list = ['pulse_interval', 'systolic_width', 'systolic_peak_time', 'diastolic_width',
                 'systolic_time', 'diastolic_time', 'time_delay', 'diastolic_peak_time']


def apply_mask_and_balance(df: pd.DataFrame, mask_type: str, balance: bool = False) -> pd.DataFrame:
    """
        Utility: Adjusts labels to enable 4-class, 3-class, or 2-class classification. Optionally balances dataset
    """
    if mask_type == '2-class':
        df['classification'] = df['classification'].replace({
            'moderate': 'stenosis',
            'light': 'stenosis',
            'full': 'stenosis'
        })
    elif mask_type == '3-class-a':
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


def find_pulse_rate_variability(entry: pd.Series) -> float:
    """
        Utility: Finds the pulse rate variability for a signal from a DataFrame row
    """
    recording = entry['signal']
    expected_distance = int(fs * 60 / heart_rate(recording, fs))
    peaks, _ = find_peaks(recording, distance=expected_distance)
    ibi = np.diff(peaks) / fs
    prv_std = np.std(ibi)
    return prv_std


def find_peak_amplitude(entry: pd.Series) -> float:
    """
        Utility: Finds the peak amplitude value for a signal from a DataFrame row
    """
    recording = entry['signal']
    expected_distance = int(fs * 60 / heart_rate(recording, fs))
    peaks, _ = find_peaks(recording, distance=expected_distance)
    peak_amplitudes = recording[peaks]
    return np.std(peak_amplitudes)


def find_mean(entry: pd.Series) -> float:
    """
        Utility: Finds the mean value for a signal from a DataFrame row
    """
    recording = entry['signal']
    return statistics.mean(recording)


def find_standard_deviation(entry: pd.Series) -> float:
    """
        Utility: Finds the standard deviation for a signal from a DataFrame row
    """
    recording = entry['signal']
    return statistics.stdev(recording)


def find_area_under_curve(entry: pd.Series) -> float:
    """
        Utility: Finds the area under the curve per length for a signal from a DataFrame row
    """
    recording = entry['signal']
    return sum(abs(x) for x in recording) / len(recording)


def find_root_mean_square(entry: pd.Series) -> float:
    """
        Utility: Finds the root-mean-square for a signal from a DataFrame row
    """
    recording = entry['signal']
    return np.sqrt(np.mean(recording ** 2))


def find_shape_factor(entry: pd.Series) -> float:
    """
        Utility: Finds the shape factor for a signal from a DataFrame row
    """
    recording = entry['signal']
    rms = find_root_mean_square(entry)
    mean_abs = np.mean(np.abs(recording))
    return rms / mean_abs


def find_min_max_deviation(entry: pd.Series) -> float:
    """
        Utility: Finds the min-max deviation for a signal from a DataFrame row
    """
    recording = entry['signal']
    return abs(max(recording) - min(recording))


# The point of having a reference_recording and a cycled_recording is to retain the same cycles when working with recordings of different derivatives
def split_cycles(reference_recording: np.ndarray, cycled_recording: np.ndarray) -> list:
    """
        Utility & Iterative: Identifies cycles using the referenced recording, splits the cycled_recording into cycles based on the cycles of the reference recording
    """
    hr = heart_rate(reference_recording, fs)
    if hr <= 0:
        raise ValueError("heart_rate returned <= 0")
    expected = int(fs * 60.0 / hr)
    expected = max(3, expected)
    inv = -reference_recording

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


def find_fiducial_points(cycle: np.ndarray, order: int) -> tuple[int, int, int]:
    """
        Utility: Finds the fiducial points in a cycle, depending on the order (0 derivative, 1st derivative, etc.)
    """
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
        return systolic_peak, dicrotic_notch, dicrotic_peak
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
        return u, v, w
    elif order == 2:
        # place holder for 2nd derivative
        print(2)
    elif order == 3:
        # place holder for 3rd derivative
        print(3)


def fiducial_point_features(cycle: np.ndarray, order: int) -> dict:
    """
        Utility: Estimates the fiducial point features for each cycle based on the fiducial points found
    """
    cycle_feature = {}
    on = cycle[0]
    if order == 0:
        sp_i, dn_i, dp_i = find_fiducial_points(cycle, order)
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
    elif order == 1:
        u_i, v_i, w_i = find_fiducial_points(cycle, order)
        u = cycle[u_i] if u_i is not None else None
        v = cycle[v_i] if v_i is not None else None
        w = cycle[w_i] if w_i is not None else None
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


def aggregate_fiducial_features(order: int, cycles: list[np.ndarray]) -> dict:
    """
    Iterative: Calculates and averages the fiducial points for every cycle in a recording, cutting out extreme values
    """
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
        return {}
