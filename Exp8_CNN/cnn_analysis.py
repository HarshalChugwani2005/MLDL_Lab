"""
Experiment 8: Convolutional Neural Network (CNN) on Fashion-MNIST
Dataset source: https://github.com/zalandoresearch/fashion-mnist

This script trains and evaluates a CNN on the Fashion-MNIST image dataset.
"""

import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Set deterministic seeds where possible for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_cnn_model(input_shape, n_classes):
    """Define a compact CNN architecture for Fashion-MNIST."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history, save_path):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("CNN Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("CNN Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sample_predictions(images, y_true, y_pred, class_names, save_path, samples=20):
    """Visualize random test samples with true/predicted labels."""
    indices = np.random.choice(len(images), size=samples, replace=False)

    plt.figure(figsize=(16, 10))
    for i, idx in enumerate(indices):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap="gray")
        true_lbl = class_names[y_true[idx]]
        pred_lbl = class_names[y_pred[idx]]
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        plt.title(f"T: {true_lbl}\nP: {pred_lbl}", color=color, fontsize=9)
        plt.axis("off")

    plt.suptitle("CNN Predictions on Fashion-MNIST Test Images", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(script_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    set_seed(42)

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    print("=" * 70)
    print("EXPERIMENT 8: CNN ON FASHION-MNIST")
    print("Dataset source: https://github.com/zalandoresearch/fashion-mnist")
    print("=" * 70)

    print("\n1. LOADING DATA")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print(f"Train shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test shape : {X_test.shape}, Labels: {y_test.shape}")

    print("\n2. PREPROCESSING")
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    print(f"Normalized and reshaped train: {X_train.shape}")
    print(f"Normalized and reshaped test : {X_test.shape}")

    print("\n3. BUILDING CNN MODEL")
    model = build_cnn_model(input_shape=(28, 28, 1), n_classes=10)
    model.summary(print_fn=lambda line: print("   " + line))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    print("\n4. TRAINING CNN")
    history = model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"Training epochs completed: {len(history.history['loss'])}")

    print("\n5. EVALUATING MODEL")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    report_text = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report_text)

    print("\n6. SAVING MODEL, METRICS, AND VISUALIZATIONS")
    model_path = os.path.join(script_dir, "fashion_mnist_cnn_model.keras")
    history_plot_path = os.path.join(viz_dir, "cnn_training_history.png")
    confusion_plot_path = os.path.join(viz_dir, "cnn_confusion_matrix.png")
    prediction_plot_path = os.path.join(viz_dir, "cnn_sample_predictions.png")
    metrics_path = os.path.join(script_dir, "cnn_metrics.txt")

    model.save(model_path)
    plot_training_history(history, history_plot_path)
    plot_confusion_matrix(y_test, y_pred, class_names, confusion_plot_path)
    plot_sample_predictions(X_test, y_test, y_pred, class_names, prediction_plot_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Experiment 8: CNN on Fashion-MNIST\n")
        f.write("Dataset source: https://github.com/zalandoresearch/fashion-mnist\n\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)

    print(f"Saved: {model_path}")
    print(f"Saved: {history_plot_path}")
    print(f"Saved: {confusion_plot_path}")
    print(f"Saved: {prediction_plot_path}")
    print(f"Saved: {metrics_path}")

    print("\nSUMMARY")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    print("EXPERIMENT 8 COMPLETED!")


if __name__ == "__main__":
    main()
