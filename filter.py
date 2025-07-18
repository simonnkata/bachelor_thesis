from scipy.signal import cheby2, filtfilt
import pandas as pd


def low_pass_filter(df, fs, cutoff, order, rs, rp):
    norm = cutoff / (0.5 * fs)
    b, a = cheby2(order, rs, norm, btype='low')
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})


def high_pass_filter(df, fs, cutoff, order, rs, rp):
    norm = cutoff / (0.5 * fs)
    b, a = cheby2(order, rs, norm, btype='high')
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})


def band_pass_filter(df, fs, low_cutoff, hight_cutoff, order, rs, rp):
    Wn = [low_cutoff / (0.5 * fs), hight_cutoff / (0.5 * fs)]
    b, a = cheby2(order, rs, Wn, btype='bandpass')
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})


def band_stop_filter(df, fs, low_cutoff, hight_cutoff, order, rs, rp):
    Wn = [low_cutoff / (0.5 * fs), hight_cutoff / (0.5 * fs)]
    b, a = cheby2(order, rs, Wn, btype='bandstop')
    return pd.DataFrame({c: filtfilt(b, a, df[c]) for c in ['R', 'G', 'B']})


def general_filter(df, fs, cutoff, order, rs, rp, high, low, filter_type):
    match filter_type:
        case 'low-pass':
            return low_pass_filter(df, fs, cutoff, order, rs, rp)
        case 'high-pass':
            return high_pass_filter(df, fs, cutoff, order, rs, rp)
        case 'band-pass':
            return band_pass_filter(df, fs, low, high, order, rs, rp)
        case 'band-stop':
            return band_stop_filter(df, fs, low, high, order, rs, rp)
