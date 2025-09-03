import pickle
from pycaret.classification import *
import numpy as np
import pandas as pd
from validation import validate
from features import extract, feature_embedding
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, cross_val_predict, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)
    print(X.head())
    y = label_df

    logo = LeaveOneGroupOut()
    clf = MLPClassifier(hidden_layer_sizes=(13,), activation="relu", solver="lbfgs", max_iter=300, random_state=42)
    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=logo)
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))


def nn_approach():
    signal_df = validate()
    signal_df = signal_df.replace({None: np.nan})
    df = extract(signal_df)
    label_df = df['classification']
    features_df = df.drop(['classification', 'patient_id'], axis=1)

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)
    print(X.head())
    y = label_df

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


def leave_one_out_approach_b():
    signal_df = validate()
    signal_df = signal_df.replace({None: np.nan})
    df = feature_embedding(signal_df)
    label_df = df['classification']
    groups = df['patient_id']
    features_df = df.drop(['classification', 'patient_id'], axis=1)

    imputer = SimpleImputer(strategy="mean")
    pipe = Pipeline([
        ("imputer", imputer),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(random_state=42, early_stopping=True, max_iter=1000))
    ])
    param_grid = {
        "clf__hidden_layer_sizes": [(64, 32), (128, 64, 32)],
        "clf__alpha": [1e-4, 1e-3],
        "clf__solver": ["adam", "lbfgs"],
        "clf__learning_rate_init": [1e-3, 1e-4]
    }

    X = features_df  # keep features_df unchanged
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

    # IMPORTANT: pass groups to .fit
    gs.fit(X, y, groups=groups)

    print("best params:", gs.best_params_)
    print("best cv score:", gs.best_score_)

    # Get cross-validated predictions using the tuned hyperparams
    best_pipe = gs.best_estimator_
    y_pred = cross_val_predict(best_pipe, X, y, groups=groups, cv=logo, n_jobs=-1)

    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()


leave_one_out_approach_b()
