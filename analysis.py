import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from filter import general_filter
import statistics
import copy
import pickle
from preprocessing import classes

with open('windowed_df.pkl', 'rb') as file:
    df = pickle.load(file)

for index, entry in df.iterrows():
    plt.clf()
    # plt.plot(df['R'], color='red', label='R')
    # plt.plot(df['G'], color='green', label='G')
    # plt.plot(df['B'], color='blue', label='B')
    plt.plot(entry['signal'])
    plt.title('Class: ' + classes[entry['class']])
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/chunks/' + str(index))
