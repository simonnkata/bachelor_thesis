from scipy.signal import detrend
from scipy.fft import rfft, rfftfreq
import numpy as np
import os
import pickle
from preprocessing import load_pos_filter_plot
import re
import matplotlib.pyplot as plt
import pandas as pd
from Validator.validator_helpers import estimate, load_ppg_struct, load_rppg_structs
from DataLoader.dataloader_helpers import Subjects_nd

classes = {
    3: 'baseline',
    4: 'full',
    5: 'moderate',
    6: 'light'
}


# Compares the estimated heart rates of the rppg and ppg signals. Returns validated rppg signals cut into 10 or 5 seconds snippets.
def validate_and_split(rppg_data: Subjects_nd, adjust_baseline_size: bool = False, output_metrics: bool = False) -> pd.DataFrame:
    """
        Compares the estimated heart rates of the rPPG and PPG signals.

        Returns:
            pd.DataFrame: Validated rPPG signals cut into 10 or 5 second snippets.
    """
    output = pd.DataFrame(columns=["signal", "classification", "patient_id"])
    ppg_structure = load_ppg_struct(adjust_baseline_size)
    rppg_structure, signal_structure, classification_structure, patient_id_structure = load_rppg_structs(rppg_data,
                                                                                                         adjust_baseline_size)
    common_keys = sorted(rppg_structure.keys() & ppg_structure.keys())
    errors = []
    for key in common_keys:
        error = abs(rppg_structure[key] - ppg_structure[key])
        errors.append(error)
        if classification_structure[key] == 'full' or error < 10:
            output.loc[len(output)] = [signal_structure[key], classification_structure[key], patient_id_structure[key]]
    if output_metrics:
        print(output['classification'].value_counts())
        print(f'Good recordings: {len(output)}')
        print(f'Bad recordings: {len(common_keys) - len(output)}')
        print(f"mean: {np.mean(errors)}")
        print(f"std: {np.std(errors)}")
        print(f"max: {np.max(errors)}")
        print(f"min: {np.min(errors)}")
        print(f"median: {np.median(errors)}")
        plt.hist(errors, bins=range(int(min(errors)), int(max(errors)) + 2), edgecolor='black')
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.show()
    return output
