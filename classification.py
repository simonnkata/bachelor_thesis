from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from validation import validate
from features import extract

signal_df = validate()
signal_df = signal_df.replace({None: np.nan})
df = extract(signal_df)


def encode_classification(x):
    if x == 'baseline':
        return 0
    elif x == 'light':
        return 1
    elif x == 'moderate':
        return 2
    else:
        return 3


print(df['classification'].value_counts())
df['classification'] = df['classification'].apply(encode_classification)
feature = df.drop('classification', axis=1)
target = df['classification']
X_train, X_test, y_train, y_test = train_test_split(feature, target, shuffle=True, test_size=0.2, random_state=1)

print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)
