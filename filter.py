from scipy.signal import cheby2, butter, cheby1, ellip, bessel, firwin, filtfilt
import pandas as pd
import numpy as np

def cheby2_filter(fs, cutoff, order, rs, df):
    norm = cutoff / (0.5 * fs)
    b, a = cheby2(order, rs, norm, btype='low')
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})

def butter_filter(fs, cutoff, order, rs, df):
    norm = cutoff / (0.5 * fs)
    b, a = butter(order, norm, btype='low')
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})

def cheby1_filter(fs, cutoff, order, rs, df):
    norm = cutoff / (0.5 * fs)
    b, a = cheby1(order, 0.5, norm, btype='low')  # using rs as rp
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})

def ellip_filter(fs, cutoff, order, rs, df):
    norm = cutoff / (0.5 * fs)
    b, a = ellip(order, 0.5, rs, norm, btype='low')  # using rs for both rp and rs
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})

def bessel_filter(fs, cutoff, order, rs, df):
    norm = cutoff / (0.5 * fs)
    b, a = bessel(order, norm, btype='low', norm='phase')
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})

def fir_filter(fs, cutoff, order, rs, df):
    norm = cutoff / (0.5 * fs)
    b = firwin(numtaps=order + 1, cutoff=norm)
    a = 1
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})