import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),])

        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(300, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def predict(model, data: tf.Tensor, threshold: np.floating) -> tuple[tf.Tensor, tf.Tensor]:
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold), loss

def iteration(df: pd.DataFrame, patient_id: int) -> tuple:
    train_df = df[df['patient_id'] != patient_id]
    test_df = df[df['patient_id'] == patient_id]
    train_data = np.stack(train_df['signal'].to_numpy())
    test_data = np.stack(test_df['signal'].to_numpy())
    train_labels = train_df['classification'].values
    test_labels = test_df['classification'].values

    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[~train_labels]
    normal_test_data  = test_data[~test_labels]


    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer='adam', loss='mae')
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )
    autoencoder.fit(normal_train_data, normal_train_data, epochs=1000, batch_size=32,
                    validation_data=(normal_test_data, normal_test_data), shuffle=True,
                    callbacks=[early_stop], verbose=0)

    reconstructions = autoencoder.predict(normal_train_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

    threshold = np.percentile(train_loss, 80)

    print("Threshold: ", threshold)
    predictions, loss = predict(autoencoder, test_data, threshold)
    return predictions, test_labels, loss

def logo_cross_validation(df: pd.DataFrame) -> None:
    patient_ids = df['patient_id'].unique()
    all_predictions = []
    all_labels = []
    all_loss = []
    for patient_id in patient_ids:
        print(f'Iterating patient {patient_id}')
        new_predictions, new_labels, new_loss = iteration(df, patient_id)
        all_predictions.extend(new_predictions)
        all_labels.extend(new_labels)
        all_loss.extend(new_loss)
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'Accuracy = {accuracy_score(all_labels, all_predictions)}')
    print(f'Precision = {precision_score(all_labels, all_predictions)}')
    print(f'Recall = {recall_score(all_labels, all_predictions)}')
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    losses = np.array(all_loss)
    labels = np.array(all_labels)

    plt.figure(figsize=(8, 5))
    sns.stripplot(x=labels, y=losses, jitter=True, alpha=0.7, palette="coolwarm")
    plt.xlabel("True label")
    plt.ylabel("Reconstruction loss")
    plt.title("Loss distribution per label")
    plt.show()