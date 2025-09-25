import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, \
    recall_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector


def logo_training_evaluation(df: pd.DataFrame, classifier: MLPClassifier, selected_features: list = None) -> None:
    """
    Trains a MLP Classifier to assign labels. Implements leave one group out validation.
    """
    label_df = df['classification']
    groups = df['patient_id']
    features_df = df.drop(['classification', 'patient_id'], axis=1)
    if selected_features is not None:
        features_df = features_df[selected_features]

    le = LabelEncoder()
    logo = LeaveOneGroupOut()
    y = le.fit_transform(label_df)
    class_names = le.classes_
    imputer = SimpleImputer(strategy="mean")
    x = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)

    pipe = Pipeline([
        ("imputer", imputer),
        ("scaler", StandardScaler()),
        ("clf", classifier)
    ])

    y_predictions = cross_val_predict(pipe, x, y, groups=groups, cv=logo)

    for i, name in enumerate(le.classes_):
        print(i, "->", name)
    print("Accuracy:", accuracy_score(y, y_predictions))
    print("Recall:", recall_score(y, y_predictions, average="macro"))
    print("F1 Score:", f1_score(y, y_predictions, average="macro"))
    print(classification_report(y, y_predictions))
    cm = confusion_matrix(y, y_predictions)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()
    plt.show()


def find_optimal_model_structure(df: pd.DataFrame, param_grid: dict, scoring: str = 'f1_macro', selected_features: list = None) -> dict:
    """
    Finds the optimal parameters for a model's structure. Eg the model size, alpha, etc.
    :return: Optimal parameters as a dict
    """
    label_df = df['classification']
    groups = df['patient_id']
    features_df = df.drop(['classification', 'patient_id'], axis=1)
    if selected_features is not None:
        features_df = features_df[selected_features]

    le = LabelEncoder()
    logo = LeaveOneGroupOut()
    scaler = StandardScaler()
    y = le.fit_transform(label_df)
    class_names = le.classes_
    imputer = SimpleImputer(strategy="mean")
    x = features_df

    pipe = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("clf", MLPClassifier(random_state=42, early_stopping=True, max_iter=10000))
    ])

    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=logo,
        scoring=scoring,
        n_jobs=-1,
        refit=True
    )

    gs.fit(x, y, groups=groups)

    print("best params:", gs.best_params_)
    print("best cv score:", gs.best_score_)

    best_pipe = gs.best_estimator_
    y_predictions = cross_val_predict(best_pipe, x, y, groups=groups, cv=logo, n_jobs=-1)

    print("Accuracy:", accuracy_score(y, y_predictions))
    print(classification_report(y, y_predictions))

    for i, name in enumerate(le.classes_):
        print(i, "->", name)
    print("Accuracy:", accuracy_score(y, y_predictions))
    print("Recall:", recall_score(y, y_predictions, average="macro"))
    print("F1 Score:", f1_score(y, y_predictions, average="macro"))
    print(classification_report(y, y_predictions))
    cm = confusion_matrix(y, y_predictions)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()
    plt.show()

    return gs.best_params_


def feature_selection(df: pd.DataFrame, classifier: MLPClassifier, number_of_features: int = None,
                      selection_type: str = 'sequential', scoring: str = 'f1_macro') -> ExhaustiveFeatureSelector | SequentialFeatureSelector:
    """
    Determines optimal features. Has options for exhaustive or sequential selection.
    :return: Selector Object
    """
    groups = df['patient_id']
    label_df = df['classification']
    features_df = df.drop(['classification', 'patient_id'], axis=1)

    le = LabelEncoder()
    logo = LeaveOneGroupOut()
    scaler = StandardScaler()
    y = le.fit_transform(label_df)
    imputer = SimpleImputer(strategy="mean")
    x = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)

    pipe = Pipeline([
        ("scaler", scaler),
        ("clf", classifier)
    ])

    if not number_of_features:
        if selection_type == 'sequential':
            number_of_features = "best"
        else:
            number_of_features = x.shape[1]

    if selection_type == 'exhaustive':
        efs = ExhaustiveFeatureSelector(
            estimator=pipe,
            min_features=1,
            max_features=number_of_features,
            scoring=scoring,
            cv=logo,
            n_jobs=-1
        )
        efs.fit(x, y, groups=groups)
        return efs
    else:
        sfs = SequentialFeatureSelector(
            pipe,
            k_features=number_of_features,
            forward=True,
            floating=False,
            scoring=scoring,
            cv=logo,
            n_jobs=-1
        )
        sfs.fit(x, y, groups=groups)
        return sfs

