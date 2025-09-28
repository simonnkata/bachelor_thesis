import pickle
from pycaret.classification import *
import numpy as np
import pandas as pd
from validation import validate
from features import extract, feature_embedding
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, cross_val_predict, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score, f1_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.feature_selection import SequentialFeatureSelector


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
    del features_df['patient_id']
    print(f'Length: {len(features_df)}')
    s = setup(features_df, target='classification', session_id=123)
    best = compare_models()
    evaluate_model(best)
    conf_df = pd.DataFrame(s._display_container[0])
    target_mapping = conf_df[conf_df['Description'] == 'Target mapping']['Value'].values
    print(target_mapping)
    plot_model(best, plot='confusion_matrix')


def leave_one_out_approach():
    signal_df = validate()
    signal_df = signal_df.replace({None: np.nan})

    df = extract(signal_df)
    label_df = df['classification']
    groups = df['patient_id']
    features_df = df.drop(['classification', 'patient_id'], axis=1)
    #selected_features = ['peak_amplitude', 'mean', 'area_under_curve', 'pulse_rate', 'systolic_width', 'time_delay', 'v_amplitude', 'w_amplitude', 'u_w_amplitude_ratio', 't_v_w', 't_u_w', 'slope_v_w', 'slope_u_w']
    selected_features = ['peak_amplitude', 'mean', 'area_under_curve', 'systolic_peak_time', 'v_amplitude', 'w_amplitude', 'slope_u_v', 'slope_v_w']
    features_df = df[selected_features]
    le = LabelEncoder()
    y = le.fit_transform(label_df)
    class_names = le.classes_

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)
    print(X.head())
    #y = label_df

    logo = LeaveOneGroupOut()
    #         ("clf", MLPClassifier(random_state=42, early_stopping=True, max_iter=10000, activation="relu",
    #                               alpha=0.01, hidden_layer_sizes=(64, 32), learning_rate_init=0.005, solver="adam"))
    # clf = MLPClassifier(hidden_layer_sizes=(13,), activation="relu", solver="lbfgs", max_iter=300, random_state=42)
    clf_1 = MLPClassifier(random_state=42, early_stopping=True, max_iter=10000, activation="tanh",
                          alpha=0.01, hidden_layer_sizes=(64, 32, 16), learning_rate_init=0.002, solver="adam")
    clf_2 = MLPClassifier(random_state=42, early_stopping=True, max_iter=10000, activation="relu",
                          alpha=0.01, batch_size=256, hidden_layer_sizes=(64, 32), learning_rate_init=0.005,
                          solver="adam")
    clf_3 = MLPClassifier(random_state=42, early_stopping=True, max_iter=10000, activation="relu",
                          alpha=0.0001, hidden_layer_sizes=(128, 64, 32), learning_rate_init=0.005,
                          solver="adam")
    pipe = Pipeline([
        ("imputer", imputer),
        ("scaler", StandardScaler()),
        ("clf", clf_2)
    ])

    y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=logo)
    # y_pred = cross_val_predict(pipe, X, y)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("Recall (macro):", recall_score(y, y_pred, average="macro"))
    print("F1 Score (macro):", f1_score(y, y_pred, average="macro"))
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()
    plt.show()


def nn_approach():
    signal_df = validate()
    signal_df = signal_df.replace({None: np.nan})
    df = extract(signal_df)
    label_df = df['classification']
    groups = df['patient_id']
    # features_df = df.drop(['classification', 'patient_id'], axis=1)
    selected_features = ['peak_amplitude', 'mean', 'area_under_curve', 'pulse_rate', 'systolic_width', 'time_delay',
                         'v_amplitude', 'w_amplitude', 'u_w_amplitude_ratio', 't_v_w', 't_u_w', 'slope_v_w',
                         'slope_u_w']
    selected_features = ['peak_amplitude', 'mean', 'area_under_curve', 'systolic_peak_time', 'v_amplitude', 'w_amplitude', 'slope_u_v',
     'slope_v_w']

    for feature in selected_features:
        current_selection = [feat for feat in selected_features if feat != feature]
        features_df = df[feature]

        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)
        print(X.head())
        y = label_df

        logo = LeaveOneGroupOut()
        # clf = MLPClassifier(hidden_layer_sizes=(13,), activation="relu", solver="lbfgs", max_iter=300, random_state=42)
        pipe = Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(random_state=42, early_stopping=True, max_iter=10000, activation="relu",
                                  alpha=0.01, batch_size=256, hidden_layer_sizes=(64, 32), learning_rate_init=0.005,
                                  solver="adam"))
        ])
        y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=logo)
        # y_pred = cross_val_predict(pipe, X, y)
        print(f"We kept  {feature}")
        print("Accuracy:", accuracy_score(y, y_pred))
        print(classification_report(y, y_pred))


def leave_one_out_approach_b():
    signal_df = validate()
    signal_df = signal_df.replace({None: np.nan})
    df = extract(signal_df)
    label_df = df['classification']
    groups = df['patient_id']
    features_df = df.drop(['classification', 'patient_id'], axis=1)

    imputer = SimpleImputer(strategy="mean")
    pipe = Pipeline([
        ("imputer", imputer),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(random_state=42, early_stopping=True, max_iter=10000))
    ])
    param_grid = {
        "clf__hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16), (128, 64, 32)],
        "clf__alpha": [1e-4, 5e-4, 1e-3, 1e-2],
        "clf__activation": ['relu', 'tanh'],
        "clf__solver": ["adam", "lbfgs"],
        "clf__learning_rate_init": [1e-4, 5e-4, 2e-3, 5e-3],
    }

    X = features_df
    y = label_df
    logo = LeaveOneGroupOut()

    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=logo,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True
    )

    gs.fit(X, y, groups=groups)

    print("best params:", gs.best_params_)
    print("best cv score:", gs.best_score_)

    #best_pipe = gs.best_estimator_
    #y_pred = cross_val_predict(best_pipe, X, y, groups=groups, cv=logo, n_jobs=-1)

    #print("Accuracy:", accuracy_score(y, y_pred))
    #print(classification_report(y, y_pred))

    #cm = confusion_matrix(y, y_pred)
    #ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    #plt.show()


def feature_selection():
    signal_df = validate()
    signal_df = signal_df.replace({None: np.nan})
    df = extract(signal_df)
    groups = df['patient_id']
    label_df = df['classification']
    features_df = df.drop(['classification', 'patient_id'], axis=1)
    clf_2 = MLPClassifier(random_state=42, early_stopping=True, max_iter=10000, activation="relu",
                          alpha=0.01, batch_size=256, hidden_layer_sizes=(64, 32), learning_rate_init=0.005,
                          solver="adam")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(random_state=42, early_stopping=True, max_iter=10000, activation="relu",
                          alpha=0.01, batch_size=256, hidden_layer_sizes=(64, 32), learning_rate_init=0.005,
                          solver="adam"))
    ])
    # batch was 256
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)
    y = label_df
    logo = LeaveOneGroupOut()

    sfs = SequentialFeatureSelector(
        pipe,
        k_features=10,
        forward=True,
        floating=False,
        scoring="f1_macro",
        cv=logo,
        n_jobs=-1
    )

    sfs.fit(X, y, groups=groups)
    selected_features = sfs.k_feature_idx_
    #print(sfs.subsets_)
    print(sfs.k_feature_names_)
    results = pd.DataFrame.from_dict(sfs.subsets_, orient="index")
    with open("i_tire.pkl", "wb") as f:
        pickle.dump(sfs, f)
    # Keep only what’s useful
    results = results[["feature_names", "avg_score"]]

    print(results)
    '''
    print(f"Number of features selected: {selected_features.sum()}")
    print(f"Selected feature indices: {np.where(selected_features)[0]}")

    # If you have feature names (DataFrame):
    if hasattr(X, 'columns'):
        selected_feature_names = X.columns[selected_features]
        print(f"Selected features: {list(selected_feature_names)}")
    '''

feature_selection()
#clf_2 is the best
# Best result: best params: {'clf__activation': 'relu', 'clf__alpha': 0.01, 'clf__batch_size': 256, 'clf__hidden_layer_sizes': (64, 32), 'clf__learning_rate_init': 0.005, 'clf__solver': 'adam'}