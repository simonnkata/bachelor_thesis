from scipy.signal import detrend
from scipy.fft import rfft, rfftfreq
import numpy as np
import os
import re
from DataLoader.dataloader_helpers import Subjects_nd
from typing import List, Dict

number_mapping = {
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g'
}

classes = {
    3: 'baseline',
    4: 'full',
    5: 'moderate',
    6: 'light'
}


# Utility: Estimates the heart rate from a recording
def heart_rate(recording: np.ndarray, fs: int) -> float:
    if len(recording) == 0:
        return 75
    recording = detrend(recording)
    length = len(recording)
    yf = np.abs(rfft(recording))
    xf = rfftfreq(length, 1 / fs)

    mask = (xf >= 0.8) & (xf <= 3)
    peak_freq = xf[mask][np.argmax(yf[mask])]
    heart_rate_bpm = peak_freq * 60
    return float(heart_rate_bpm)


# Iterative: Iterates through Subjects_nd element, estimates heart rates
def estimate(_data: Subjects_nd) -> List[List[float]]:
    fs = 30
    results = []
    for subject in _data:
        subject_data = []
        for recording in subject:
            heart_rate_bpm = heart_rate(recording, fs)
            subject_data.append(heart_rate_bpm)
        results.append(subject_data)
    return results


# Utility & Iterative: Loads a ppg recording, splits the recording into snippets, returns dict of snippet_id and estimated heart rate
def load_and_split_ppg(filename: str, subject: str, subject_number: int, adjust_baseline_size: bool, fs: int) -> Dict:
    iter_ppg_structure = {}
    recording_type = None
    if filename.endswith('.txt'):
        file_path = os.path.join('_processed/' + subject, filename)
        match = re.match(r"\d+([a-zA-Z])(.*)", filename)
        if match:
            recording_type = match.group(1)
        recording = np.loadtxt(file_path, usecols=1)
        # Work only with recordings of the type d,e,f,g
        if recording_type in ['d', 'e', 'f', 'g']:
            length, snippet_length = 120, 10
            if adjust_baseline_size:
                length = 60 if recording_type == 'd' else 120
                snippet_length = 5 if recording_type == 'd' else 10
            step = fs * snippet_length
            for i in range(0, length * fs, step):
                window = recording[i:i + step]
                if len(window) < step:
                    break
                estimated_heart_rate = heart_rate(window, fs)
                iter_ppg_structure[
                    f'{subject_number + 8}_{recording_type}_{i / step}'] = estimated_heart_rate
    return iter_ppg_structure


# Iterative: Iterates through _processed directory, applies load_and_split_ppg to every recording.
def load_ppg_struct(adjust_baseline_size: bool = False) -> Dict:
    ppg_structure = {}
    _data = []
    fs = 128
    files = sorted([f for f in os.listdir('_processed') if f.isdigit()], key=int)
    for subject_number, subject in enumerate(files):
        for recording_number, filename in enumerate(os.listdir('_processed/' + subject)):
            iter_ppg_structure = load_and_split_ppg(filename, subject, subject_number, adjust_baseline_size, fs)
            ppg_structure.update(iter_ppg_structure)
    return ppg_structure


# Utility & Iterative: Splits one rPPG recording into snippets, returns dict of snippet_id and estimated heart rate, classification, patient_id, and snippet
def load_and_split_rppg(subject: List[np.ndarray], subject_number: int, recording_number: int, adjust_baseline_size: bool, fs: int) -> tuple[Dict, Dict, Dict, Dict]:
    iter_rppg_structure = {}
    iter_signal_structure = {}
    iter_classification_structure = {}
    iter_patient_id_structure = {}
    recording = subject[recording_number]
    length, snippet_length = 120, 10
    if adjust_baseline_size:
        length = 60 if recording_number == 3 else 120
        snippet_length = 5 if recording_number == 3 else 10
    step = fs * snippet_length
    for i in range(0, length * fs, step):
        window = recording[i:i + step]
        if len(window) < step:
            break
        estimated_heart_rate = heart_rate(recording[i:i + step], fs)
        iter_rppg_structure[
            f'{subject_number}_{number_mapping[recording_number]}_{i / (fs * snippet_length)}'] = estimated_heart_rate
        iter_signal_structure[
            f'{subject_number}_{number_mapping[recording_number]}_{i / (fs * snippet_length)}'] = window
        iter_classification_structure[
            f'{subject_number}_{number_mapping[recording_number]}_{i / (fs * snippet_length)}'] = classes[
            recording_number]
        iter_patient_id_structure[
            f'{subject_number}_{number_mapping[recording_number]}_{i / (fs * snippet_length)}'] = subject_number
    return iter_rppg_structure, iter_signal_structure, iter_classification_structure, iter_patient_id_structure


# Iterative: Iterates through all rppg recordings, applies load_and_split_rppg to every recording.
def load_rppg_structs(rppg_data: Subjects_nd, adjust_baseline_size: bool) -> tuple[Dict, Dict, Dict, Dict]:
    rppg_structure = {}
    signal_structure = {}
    classification_structure = {}
    patient_id_structure = {}
    fs = 30
    for subject_number in range(8, len(rppg_data)):
        subject = rppg_data[subject_number]
        for recording_number in range(3, len(subject)):
            (new_rppg_structure, new_signal_structure, new_classification_structure,
             new_patient_id_structure) = load_and_split_rppg(subject, subject_number, recording_number, adjust_baseline_size, fs)
            rppg_structure.update(new_rppg_structure)
            signal_structure.update(new_signal_structure)
            classification_structure.update(new_classification_structure)
            patient_id_structure.update(new_patient_id_structure)
    return rppg_structure, signal_structure, classification_structure, patient_id_structure
