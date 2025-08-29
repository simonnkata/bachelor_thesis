import pickle
from pycaret.classification import *
import numpy as np
import pandas as pd
from validation import validate
from features import extract


def pycaret_version():
    df = validate()
    class_counts = df['classification'].value_counts()
    min_count = class_counts.min()

    # Now, sample each class to the size of the smallest one
    balanced_df = (
        df.groupby('classification', group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
    )
    df = df.replace({None: np.nan})
    features_df = extract(df)
    print(f'Length: {len(features_df)}')
    s = setup(features_df, target='classification', session_id=123)
    best = compare_models()
    evaluate_model(best)
    conf_df = pd.DataFrame(s._display_container[0])
    target_mapping = conf_df[conf_df['Description'] == 'Target mapping']['Value'].values
    print(target_mapping)
    plot_model(best, plot='confusion_matrix')


pycaret_version()
