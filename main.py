import numpy as np
from sklearn.base import clone
from DataLoader.dataloader import load_process_data
from FeatureExtraction.feature_extractor import extract
from Validator.validator import validate_and_split
from Models.mlp_classifier import logo_training_evaluation, find_optimal_model_structure, feature_selection
from sklearn.neural_network import MLPClassifier
from Models.transformer import logo_cross_validation

# THIS SECTION IS FOR MLP Classifier
# Prepare the data. Here can be adjusted for 4-class, 3-class, 3-class-b, or 2-class.
loaded_data = load_process_data()
dataframe = validate_and_split(loaded_data, True)
dataframe = dataframe.replace({None: np.nan})
features_df = extract(dataframe, '4-class', False)

# Determine the best model structure.
main_param_grid = {
    "clf__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (64, 32, 16), (128, 64, 32), (100, 50, 25)],
    "clf__alpha": [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "clf__activation": ['relu', 'tanh'],
    "clf__solver": ["adam", "lbfgs"],
    "clf__learning_rate_init": [1e-3, 1e-4, 5e-4, 2e-3, 5e-3]
}

smaller_param_grid = {
    "clf__hidden_layer_sizes": [(128, 64, 32), (100, 50, 25)],
    "clf__alpha": [1e-2],
    "clf__activation": ['relu'],
    "clf__solver": ["adam", "lbfgs"],
    "clf__learning_rate_init": [2e-3, 5e-3]
}

best_params = find_optimal_model_structure(features_df, main_param_grid)

classifier = MLPClassifier(
    hidden_layer_sizes=best_params['clf__hidden_layer_sizes'],
    activation=best_params['clf__activation'],
    solver=best_params['clf__solver'],
    learning_rate_init=best_params['clf__learning_rate_init'],
    alpha=best_params['clf__alpha'],
    max_iter=10000,
    random_state=42,
    early_stopping=True
)

# Select best features
selected_features = list(feature_selection(features_df, clone(classifier), selection_type='sequential').k_feature_names_)
print(selected_features)

# Evaluate with best model structure, and best features
logo_training_evaluation(features_df, classifier, selected_features)

# THIS SECTION IS FOR TRANSFORMER
loaded_data_2 = load_process_data()
transformer_df = validate_and_split(loaded_data_2, False)
transformer_df = transformer_df[transformer_df['classification'].isin(['baseline', 'full'])]
transformer_df['classification'] = transformer_df['classification'].apply(
    lambda x: 0 if x == 'baseline' else 1
)
logo_cross_validation(transformer_df)

