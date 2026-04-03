"""
Experiment 7: Artificial Neural Network (ANN) for Churn Prediction
Dataset: Churn_Modelling.csv

This script builds and evaluates a Keras/TensorFlow ANN for binary
classification of customer churn.
"""

import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_dim):
    """Build the ANN model for churn classification."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def plot_training_history(history, save_path):
    """Plot training history curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_title("ANN Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    axes[1].set_title("ANN Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Churn Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ANN ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(script_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    set_seed(42)

    print("=" * 70)
    print("EXPERIMENT 7: ARTIFICIAL NEURAL NETWORK (ANN)")
    print("Dataset: Churn_Modelling.csv")
    print("=" * 70)

    print("\n1. LOADING DATA")
    dataset_path = os.path.join(script_dir, "Churn_Modelling.csv")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Churn distribution:\n{df['Exited'].value_counts().sort_index()}")

    print("\n2. FEATURE SELECTION")
    # Keep informative numerical and categorical features; drop identifiers and surname.
    target_col = "Exited"
    feature_cols = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()

    categorical_cols = ["Geography", "Gender"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    print(f"Using features ({len(feature_cols)}): {feature_cols}")

    print("\n3. TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\n4. PREPROCESSING")
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape : {X_test_processed.shape}")

    print("\n5. BUILDING ANN MODEL")
    model = build_model(X_train_processed.shape[1])
    model.summary(print_fn=lambda line: print("   " + line))

    # Handle class imbalance using class weights.
    class_counts = y_train.value_counts().to_dict()
    total = len(y_train)
    class_weights = {
        cls: total / (2.0 * count) for cls, count in class_counts.items()
    }
    print(f"Class weights: {class_weights}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]

    print("\n6. TRAINING ANN")
    history = model.fit(
        X_train_processed,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    print(f"Training epochs completed: {len(history.history['loss'])}")

    print("\n7. EVALUATION")
    y_prob = model.predict(X_test_processed, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, digits=4)}")

    print("\n8. SAVING MODEL AND VISUALIZATIONS")
    model_path = os.path.join(script_dir, "ann_churn_model.keras")
    history_path = os.path.join(viz_dir, "ann_training_history.png")
    cm_path = os.path.join(viz_dir, "ann_confusion_matrix.png")
    roc_path = os.path.join(viz_dir, "ann_roc_curve.png")
    metrics_path = os.path.join(script_dir, "ann_metrics.txt")

    model.save(model_path)
    plot_training_history(history, history_path)
    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_roc_curve(y_test, y_prob, roc_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Experiment 7: ANN Churn Prediction\n")
        f.write("Dataset: Churn_Modelling.csv\n\n")
        f.write("FEATURES USED:\n")
        for feature in feature_cols:
            f.write(f"  - {feature}\n")
        f.write("\nMODEL SUMMARY:\n")
        f.write("  - Framework: TensorFlow/Keras\n")
        f.write("  - Hidden layers: 128, 64, 32\n")
        f.write("  - Activation: ReLU\n")
        f.write("  - Output activation: Sigmoid\n")
        f.write("  - Loss: Binary Crossentropy\n")
        f.write("  - Optimizer: Adam\n")
        f.write("  - Class weights: Enabled\n\n")
        f.write("PERFORMANCE:\n")
        f.write(f"  Accuracy : {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall   : {recall:.4f}\n")
        f.write(f"  F1 Score : {f1:.4f}\n")
        f.write(f"  ROC AUC  : {auc:.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_pred, digits=4))

    print(f"Saved: {model_path}")
    print(f"Saved: {history_path}")
    print(f"Saved: {cm_path}")
    print(f"Saved: {roc_path}")
    print(f"Saved: {metrics_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"Samples used : {len(df)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test F1 Score : {f1:.4f}")
    print(f"Test ROC AUC  : {auc:.4f}")



if __name__ == "__main__":
    main()
