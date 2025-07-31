# The purpose of this script is to validate the preprocessing pipeline: Here we estimate the pulse frequency from the
# reference ppg recordings, and compare that with the estimated pulse from the processed ppg recordings
from scipy.signal import detrend
from scipy.fft import rfft, rfftfreq
import numpy as np
import os
import pickle


def estimate(_data):
    fs = 30
    results = []
    for subject in _data:
        subject_data = []
        for recording in subject:
            ppg = detrend(recording)
            N = len(ppg)
            yf = np.abs(rfft(ppg))
            xf = rfftfreq(N, 1 / fs)

            mask = (xf >= 0.8) & (xf <= 3)
            peak_freq = xf[mask][np.argmax(yf[mask])]
            heart_rate_bpm = peak_freq * 60
            subject_data.append(heart_rate_bpm)
        results.append(subject_data)
    return results


with open('estimations_rppg.pkl', 'rb') as f:
    estimations_rppg = pickle.load(f)
with open('estimations_ppg.pkl', 'rb') as f:
    estimations_ppg = pickle.load(f)

del estimations_rppg[13]
del estimations_ppg[13]

errors = []
for subject_ppg, subject_rppg in zip(estimations_rppg[9:], estimations_ppg[9:]):
    for estimation_ppg, estimation_rppg in zip(subject_ppg, subject_rppg):
        if estimation_ppg is not None and estimation_rppg is not None:
            errors.append(abs(estimation_ppg - estimation_rppg))
