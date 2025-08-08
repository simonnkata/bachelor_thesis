# The purpose of this script is to validate the preprocessing pipeline: Here we estimate the pulse frequency from the
# reference ppg recordings, and compare that with the estimated pulse from the processed ppg recordings
from scipy.signal import detrend
from scipy.fft import rfft, rfftfreq
import numpy as np
import os
import pickle
from preprocessing import load_pos_filter_plot
import re
import matplotlib.pyplot as plt


def heart_rate(recording, fs):
    recording = detrend(recording)
    N = len(recording)
    yf = np.abs(rfft(recording))
    xf = rfftfreq(N, 1 / fs)

    mask = (xf >= 0.8) & (xf <= 3)
    peak_freq = xf[mask][np.argmax(yf[mask])]
    heart_rate_bpm = peak_freq * 60
    return heart_rate_bpm


def estimate(_data):
    fs = 30
    results = []
    for subject in _data:
        subject_data = []
        for recording in subject:
            heart_rate_bpm = heart_rate(recording, fs)
            subject_data.append(heart_rate_bpm)
        results.append(subject_data)
    return results


def compare_estimations():
    with open('estimations_rppg.pkl', 'rb') as f:
        estimations_rppg = pickle.load(f)
    with open('estimations_ppg.pkl', 'rb') as f:
        estimations_ppg = pickle.load(f)

    del estimations_rppg[13]
    del estimations_ppg[13]

    errors = []
    for subject_ppg, subject_rppg in zip(estimations_ppg[9:], estimations_rppg[9:]):
        for estimation_ppg, estimation_rppg in zip(subject_ppg, subject_rppg):
            if estimation_ppg is not None and estimation_rppg is not None:
                errors.append(abs(estimation_ppg - estimation_rppg))

    print(f"mean: {np.mean(errors)}")
    print(f"std: {np.std(errors)}")
    print(f"max: {np.max(errors)}")
    print(f"min: {np.min(errors)}")
    print(f"median: {np.median(errors)}")


def validate():
    number_mapping = {
        3: 'd',
        4: 'e',
        5: 'f',
        6: 'g'
    }
    # Calculate estimations for rppg
    #with open('lowkey_dont_need/normalised.pkl', 'rb') as file:
    #    rppg = pickle.load(file)
    rppg = load_pos_filter_plot()
    rppg_structure = {}
    fs = 30
    for subject_number in range(8, len(rppg)):
        subject = rppg[subject_number]
        print(f'Processing subject {subject_number}...')
        for recording_number in range(3, len(subject)):
            recording = subject[recording_number]
            length = 60 if recording_number == 3 else 120
            step = fs * 10
            for i in range(0, length * fs, step):
                # print(recording)
                window = recording[i:i + step]
                if len(window) < step:
                    break
                estimated_heart_rate = heart_rate(recording[i:i + step], fs)
                rppg_structure[
                    f'{subject_number}_{number_mapping[recording_number]}_{i / (fs * 10)}'] = estimated_heart_rate

    # Calculate estimations for ppg
    fs = 128
    ppg_structure = {}
    files = sorted([f for f in os.listdir('_processed') if f.isdigit()], key=int)
    print(files)
    for subject_number, subject in enumerate(files):
        print(f'Processing subject {subject}...')
        for recording_number, filename in enumerate(os.listdir('_processed/' + subject)):
            if filename.endswith('.txt'):
                file_path = os.path.join('_processed/' + subject, filename)
                match = re.match(r"\d+([a-zA-Z])(.*)", filename)
                if match:
                    recording_type = match.group(1)
                recording = np.loadtxt(file_path, usecols=1)
                if recording_type in ['d', 'e', 'f', 'g']:
                    length = 60 if recording_number == 1 else 120
                    for i in range(0, length * fs, fs * 10):
                        window = recording[i:i + (fs*10)]
                        if len(window) < (fs*10):
                            break
                        estimated_heart_rate = heart_rate(window, fs)
                        ppg_structure[
                            f'{subject_number}_{recording_type}_{i / (fs * 10)}'] = estimated_heart_rate
    # with open('rppg_structure.pkl', 'wb') as file:
    #    pickle.dump(rppg_structure, file)
    #with open('ppg_structure.pkl', 'wb') as file:
    #    pickle.dump(ppg_structure, file)

    common_keys = sorted(rppg_structure.keys() & ppg_structure.keys())
    errors = []
    ppg_estimates = np.array([ppg_structure[k] for k in common_keys])
    rppg_estimates = np.array([rppg_structure[k] for k in common_keys])
    mean = (ppg_estimates + rppg_estimates) / 2
    diff = ppg_estimates - rppg_estimates
    md = np.mean(diff)
    sd = np.std(diff)

    plt.scatter(mean, diff)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='red', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='red', linestyle='--')
    plt.xlabel('Mean of two estimates')
    plt.ylabel('Difference between estimates')
    plt.title('Bland-Altman Plot PPG vs RPPG')
    plt.show()

    for key in common_keys:
        error = rppg_structure[key] - ppg_structure[key]
        errors.append(abs(error))

    print(f"mean: {np.mean(errors)}")
    print(f"std: {np.std(errors)}")
    print(f"max: {np.max(errors)}")
    print(f"min: {np.min(errors)}")
    print(f"median: {np.median(errors)}")
    plt.hist(errors, bins=range(int(min(errors)), int(max(errors)) + 2), edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.show()
