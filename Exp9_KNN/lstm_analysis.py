"""
Aim: Implement Recurrent Neural Network (RNN) / LSTM for time series data.
Experiment 9: LSTM on Air Quality UCI Dataset

This script trains an LSTM model for multivariate time-series forecasting using
PT08 air-quality sensor readings.
"""

import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path, delimiter=";")

    for col in df.columns:
        if df[col].dtype == "object" and col not in ["Date", "Time"]:
            df[col] = df[col].astype(str).str.replace(",", ".")

    numeric_cols = df.columns[2:]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace(-200, np.nan)
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    return df


def make_sequences(data, lookback=24):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def build_lstm(lookback, n_features):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, n_features)),
            tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(n_features),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("LSTM Training Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["mae"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Validation MAE")
    axes[1].set_title("LSTM Training MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(y_true, y_pred, feature_names, save_path):
    n_features = y_true.shape[1]
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))

    if n_features == 1:
        axes = [axes]

    for i in range(n_features):
        axes[i].plot(y_true[:, i], label="Actual", linewidth=1)
        axes[i].plot(y_pred[:, i], label="Predicted", linewidth=1)
        axes[i].set_title(f"LSTM Prediction: {feature_names[i]}")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("Normalized Value")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, feature_names, save_path):
    residuals = y_true - y_pred
    n_features = residuals.shape[1]

    fig, axes = plt.subplots(n_features, 2, figsize=(12, 3 * n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_features):
        axes[i, 0].plot(residuals[:, i], linewidth=0.8, alpha=0.7)
        axes[i, 0].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[i, 0].set_title(f"Residuals Over Time: {feature_names[i]}")
        axes[i, 0].set_xlabel("Time Step")
        axes[i, 0].set_ylabel("Residual")
        axes[i, 0].grid(alpha=0.3)

        axes[i, 1].hist(residuals[:, i], bins=30, edgecolor="black", alpha=0.8)
        axes[i, 1].set_title(f"Residual Distribution: {feature_names[i]}")
        axes[i, 1].set_xlabel("Residual Value")
        axes[i, 1].set_ylabel("Frequency")
        axes[i, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(script_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    set_seed(42)

    print("=" * 70)
    print("EXPERIMENT 9: LSTM FOR TIME SERIES FORECASTING")
    print("Dataset: Air Quality UCI - PT08 Sensor Readings")
    print("=" * 70)

    print("\n1. LOADING AND PREPROCESSING DATA")
    csv_path = os.path.join(script_dir, "AirQualityUCI.csv")
    df = load_and_preprocess_data(csv_path)

    feature_cols = [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
    ]
    available_cols = [c for c in feature_cols if c in df.columns]

    print(f"Dataset shape: {df.shape}")
    print(f"Using features: {available_cols}")

    df_subset = df[available_cols].copy()
    print("\nMissing values BEFORE handling:")
    print(df_subset.isnull().sum())

    df_subset = df_subset.interpolate(method="linear", limit_direction="both")
    df_clean = df_subset.dropna()

    print(f"\nClean dataset shape: {df_clean.shape}")
    print(f"Rows after removing NaN: {len(df_clean)}")

    print("\n2. NORMALIZING DATA")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_clean[available_cols])
    print(f"Scaled data shape: {data_scaled.shape}")

    print("\n3. PREPARING SEQUENCES FOR LSTM")
    lookback = 24
    X, y = make_sequences(data_scaled, lookback=lookback)
    print(f"Sequences shape: X={X.shape}, y={y.shape}")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training sequences: {X_train.shape}")
    print(f"Testing sequences : {X_test.shape}")

    print("\n4. BUILDING LSTM MODEL")
    model = build_lstm(lookback=lookback, n_features=len(available_cols))
    model.summary(print_fn=lambda line: print("   " + line))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    print("\n5. TRAINING LSTM MODEL")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"Training completed in {len(history.history['loss'])} epochs")

    print("\n6. EVALUATING ON TEST SET")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)

    # Metrics for the first feature in original scale.
    y_test_feat1 = y_test[:, 0]
    y_pred_feat1 = y_pred[:, 0]

    y_test_full = np.hstack([y_test_feat1.reshape(-1, 1), np.zeros((len(y_test_feat1), len(available_cols) - 1))])
    y_pred_full = np.hstack([y_pred_feat1.reshape(-1, 1), np.zeros((len(y_pred_feat1), len(available_cols) - 1))])

    y_test_original = scaler.inverse_transform(y_test_full)[:, 0]
    y_pred_original = scaler.inverse_transform(y_pred_full)[:, 0]

    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test_original, y_pred_original)

    print(f"Test Loss (normalized): {test_loss:.6f}")
    print(f"Test MAE (normalized) : {test_mae:.6f}")
    print(f"MAE (Original scale): {mae:.2f}")
    print(f"RMSE (Original scale): {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    print("\n7. SAVING MODEL, METRICS, AND VISUALIZATIONS")
    model_path = os.path.join(script_dir, "lstm_air_quality_model.keras")
    history_path = os.path.join(viz_dir, "lstm_training_history.png")
    pred_path = os.path.join(viz_dir, "lstm_predictions_vs_actual.png")
    residual_path = os.path.join(viz_dir, "lstm_residuals.png")
    metrics_path = os.path.join(script_dir, "lstm_metrics.txt")

    model.save(model_path)
    plot_history(history, history_path)
    plot_predictions(y_test, y_pred, available_cols, pred_path)
    plot_residuals(y_test, y_pred, available_cols, residual_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Aim: Implement Recurrent Neural Network (RNN) / LSTM for time series data\n")
        f.write("Experiment 9: LSTM on Air Quality UCI Dataset\n")
        f.write("=" * 70 + "\n\n")
        f.write("Features used:\n")
        for c in available_cols:
            f.write(f"  - {c}\n")
        f.write("\nModel:\n")
        f.write("  LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16) -> Dense(5)\n")
        f.write("\nMetrics (feature PT08.S1(CO) in original scale):\n")
        f.write(f"  Test Loss (normalized): {test_loss:.6f}\n")
        f.write(f"  Test MAE (normalized): {test_mae:.6f}\n")
        f.write(f"  MAE (original): {mae:.2f}\n")
        f.write(f"  RMSE (original): {rmse:.2f}\n")
        f.write(f"  R2 Score: {r2:.4f}\n")

    print(f"Saved: {model_path}")
    print(f"Saved: {history_path}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {residual_path}")
    print(f"Saved: {metrics_path}")

    print("\nEXPERIMENT 9 COMPLETED!")


if __name__ == "__main__":
    main()
