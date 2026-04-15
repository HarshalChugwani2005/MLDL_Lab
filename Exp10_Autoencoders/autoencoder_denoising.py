"""
Experiment 10: Autoencoder for image denoising on MNIST IDX files.

This script:
1. Loads MNIST images from local IDX files.
2. Adds Gaussian noise to create noisy inputs.
3. Trains a convolutional autoencoder.
4. Evaluates denoising quality.
5. Saves model, metrics, and visualizations.
"""

import os
import random
import struct
import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_idx_path(dataset_dir: str, filename: str) -> str:
    """Resolve IDX file path from either flat or extracted folder layout."""
    direct_path = os.path.join(dataset_dir, filename)
    nested_path = os.path.join(dataset_dir, filename.replace(".idx", "-idx"), filename.replace(".idx", "-idx"))

    if os.path.exists(direct_path):
        return direct_path
    if os.path.exists(nested_path):
        return nested_path

    raise FileNotFoundError(f"Could not locate file: {filename} in {dataset_dir}")


def load_idx_images(file_path: str) -> np.ndarray:
    """Load images from an IDX image file into shape (N, 28, 28)."""
    with open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in image file: {file_path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)

    images = data.reshape(num_images, rows, cols).astype("float32") / 255.0
    return images


def add_gaussian_noise(images: np.ndarray, noise_factor: float = 0.4) -> np.ndarray:
    """Add Gaussian noise to images and clip output to [0, 1]."""
    noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0.0, 1.0).astype("float32")


def build_autoencoder(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Build a convolutional denoising autoencoder."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    outputs = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inputs, outputs, name="mnist_denoising_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def plot_training_loss(history: tf.keras.callbacks.History, save_path: str) -> None:
    """Save train/validation loss plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_denoising_examples(
    clean_images: np.ndarray,
    noisy_images: np.ndarray,
    denoised_images: np.ndarray,
    save_path: str,
    n: int = 10,
) -> None:
    """Save a grid with clean, noisy, and denoised examples."""
    n = min(n, len(clean_images))
    fig, axes = plt.subplots(3, n, figsize=(1.8 * n, 5.2))

    for i in range(n):
        axes[0, i].imshow(clean_images[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Clean", fontsize=10)

        axes[1, i].imshow(noisy_images[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Noisy", fontsize=10)

        axes[2, i].imshow(denoised_images[i].squeeze(), cmap="gray")
        axes[2, i].axis("off")
        axes[2, i].set_title("Denoised", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def compute_psnr(clean: np.ndarray, pred: np.ndarray) -> float:
    """Compute average PSNR in dB over image batch."""
    psnr_vals = tf.image.psnr(clean, pred, max_val=1.0).numpy()
    return float(np.mean(psnr_vals))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training control."""
    parser = argparse.ArgumentParser(
        description="Train a convolutional autoencoder for MNIST denoising"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--noise-factor", type=float, default=0.4, help="Gaussian noise factor")
    parser.add_argument(
        "--train-subset",
        type=int,
        default=0,
        help="Use only first N training images (0 means full dataset)",
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=0,
        help="Use only first N test images (0 means full dataset)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "Dataset")
    viz_dir = os.path.join(script_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    set_seed(42)

    print("=" * 72)
    print("EXPERIMENT 10: AUTOENCODER FOR IMAGE DENOISING")
    print("Dataset: MNIST IDX files (local)")
    print("=" * 72)

    print("\n1. LOADING DATA")
    train_images_path = resolve_idx_path(dataset_dir, "train-images.idx3-ubyte")
    test_images_path = resolve_idx_path(dataset_dir, "t10k-images.idx3-ubyte")

    x_train = load_idx_images(train_images_path)
    x_test = load_idx_images(test_images_path)

    if args.train_subset > 0:
        x_train = x_train[: args.train_subset]
    if args.test_subset > 0:
        x_test = x_test[: args.test_subset]

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Train shape: {x_train.shape}")
    print(f"Test shape : {x_test.shape}")

    print("\n2. CREATING NOISY DATA")
    noise_factor = args.noise_factor
    x_train_noisy = add_gaussian_noise(x_train, noise_factor=noise_factor)
    x_test_noisy = add_gaussian_noise(x_test, noise_factor=noise_factor)

    print(f"Noise factor: {noise_factor}")

    print("\n3. BUILDING AUTOENCODER")
    model = build_autoencoder(input_shape=(28, 28, 1))
    model.summary(print_fn=lambda line: print("   " + line))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    print("\n4. TRAINING")
    history = model.fit(
        x_train_noisy,
        x_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"Training epochs completed: {len(history.history['loss'])}")

    print("\n5. EVALUATION")
    x_test_denoised = model.predict(x_test_noisy, verbose=0)

    test_mse = float(np.mean((x_test - x_test_denoised) ** 2))
    baseline_mse = float(np.mean((x_test - x_test_noisy) ** 2))
    psnr_noisy = compute_psnr(x_test, x_test_noisy)
    psnr_denoised = compute_psnr(x_test, x_test_denoised)

    print(f"Baseline MSE (Noisy vs Clean) : {baseline_mse:.6f}")
    print(f"Autoencoder MSE (Denoised vs Clean): {test_mse:.6f}")
    print(f"PSNR Noisy    : {psnr_noisy:.3f} dB")
    print(f"PSNR Denoised : {psnr_denoised:.3f} dB")

    print("\n6. SAVING ARTIFACTS")
    model_path = os.path.join(script_dir, "autoencoder_denoising_model.keras")
    loss_plot_path = os.path.join(viz_dir, "autoencoder_training_loss.png")
    samples_plot_path = os.path.join(viz_dir, "denoising_examples.png")
    metrics_path = os.path.join(script_dir, "autoencoder_metrics.txt")

    model.save(model_path)
    plot_training_loss(history, loss_plot_path)
    plot_denoising_examples(x_test, x_test_noisy, x_test_denoised, samples_plot_path, n=10)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Experiment 10: Autoencoder Image Denoising\n")
        f.write("Dataset: MNIST IDX files\n\n")
        f.write("MODEL:\n")
        f.write("  - Type: Convolutional Autoencoder\n")
        f.write("  - Input shape: 28x28x1\n")
        f.write("  - Loss: Mean Squared Error\n")
        f.write("  - Optimizer: Adam (lr=0.001)\n")
        f.write(f"  - Epochs max: {args.epochs}\n")
        f.write(f"  - Batch size: {args.batch_size}\n")
        f.write(f"  - Noise factor: {noise_factor}\n\n")
        f.write("RESULTS:\n")
        f.write(f"  - Baseline MSE (Noisy vs Clean): {baseline_mse:.6f}\n")
        f.write(f"  - Denoised MSE (Autoencoder): {test_mse:.6f}\n")
        f.write(f"  - PSNR Noisy: {psnr_noisy:.3f} dB\n")
        f.write(f"  - PSNR Denoised: {psnr_denoised:.3f} dB\n")

    print(f"Saved: {model_path}")
    print(f"Saved: {loss_plot_path}")
    print(f"Saved: {samples_plot_path}")
    print(f"Saved: {metrics_path}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"Train samples: {x_train.shape[0]}")
    print(f"Test samples : {x_test.shape[0]}")
    print(f"Denoised MSE : {test_mse:.6f}")
    print(f"Denoised PSNR: {psnr_denoised:.3f} dB")


if __name__ == "__main__":
    main()
