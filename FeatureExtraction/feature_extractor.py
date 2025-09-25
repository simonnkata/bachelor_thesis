import pickle
import pandas as pd
import numpy as np
from FeatureExtraction.feature_extractor_helper import (find_peak_amplitude, find_mean, find_standard_deviation,
                                                        find_root_mean_square, find_min_max_deviation,
                                                        find_shape_factor, apply_mask_and_balance,
                                                        find_area_under_curve, find_pulse_rate_variability,
                                                        split_cycles, aggregate_fiducial_features)


def extract(df):
    """
        Takes a data frame with the signals, labels, and patient_id and extracts features.
        Returns:
             DataFrame with the labels, patient_id, peak_amplitude, area_under_curve, etc.
    """
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
        cycles = split_cycles(recording, recording)
        first_derivative_cycles = split_cycles(recording, first_derivative)
        fiducial_features = aggregate_fiducial_features(0, cycles)
        fiducial_features_1 = aggregate_fiducial_features(1, first_derivative_cycles)

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

    features_df = apply_mask_and_balance(features_df, '4-class', 0)
    print(f'We are working with {len(features_df)} rows')
    return features_df
