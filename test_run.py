import pickle
from pycaret.classification import *
import numpy as np
from validation import validate
from features import extract

df = validate()
df = df.replace({None: np.nan})
features_df = extract(df)

s = setup(features_df, target='classification', session_id=123)
best = compare_models()
evaluate_model(best)
plot_model(best, plot='confusion_matrix')
