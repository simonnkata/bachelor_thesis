from scipy.signal import cheby2, filtfilt
import pandas as pd


def cheby2_filter(fs, cutoff, order, rs, df):
    normalised_cutoff = cutoff / (0.5 * fs)
    b, a = cheby2(N=order, rs=rs, Wn=normalised_cutoff, btype='low', analog=False)
    filtered_df = pd.DataFrame()
    filtered_df['R'] = filtfilt(b, a, df['R'])
    filtered_df['G'] = filtfilt(b, a, df['G'])
    filtered_df['B'] = filtfilt(b, a, df['B'])
    return filtered_df
