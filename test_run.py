import pickle
from pycaret.classification import *
import numpy as np
import pandas as pd
from validation import validate
from features import extract
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


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


def nn_approach():
    signal_df = validate()
    signal_df = signal_df.replace({None: np.nan})
    df = extract(signal_df)
    label_df = df['classification']
    groups = df['patient_id']
    features_df = df.drop(['classification', 'patient_id'], axis=1)

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)
    print(X.head())
    y = label_df

    logo = LeaveOneGroupOut()
    clf = MLPClassifier(hidden_layer_sizes=(13,), activation="relu", solver="lbfgs", max_iter=300, random_state=42)
    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=logo)
    print("Accuracy:", accuracy_score(y, y_pred))

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = MLPClassifier(hidden_layer_sizes=(13,), activation="relu", solver="lbfgs", max_iter=300, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    '''

nn_approach()
