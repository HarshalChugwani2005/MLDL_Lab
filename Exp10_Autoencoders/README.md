# Experiment 10: Autoencoder for Image Denoising

This experiment implements a convolutional autoencoder for denoising MNIST images using local IDX files in the `Dataset` folder.

## What this script does

- Loads MNIST train/test images directly from IDX files.
- Adds Gaussian noise to input images.
- Trains a convolutional autoencoder to reconstruct clean images.
- Evaluates denoising quality using:
  - MSE (lower is better)
  - PSNR in dB (higher is better)
- Saves:
  - Trained model (`autoencoder_denoising_model.keras`)
  - Training loss plot (`visualizations/autoencoder_training_loss.png`)
  - Sample denoising grid (`visualizations/denoising_examples.png`)
  - Metrics report (`autoencoder_metrics.txt`)

## Script

- `autoencoder_denoising.py`

## Run commands

From workspace root:

```powershell
& "c:/Users/harsh/MLDL LAB/.venv/Scripts/python.exe" "c:/Users/harsh/MLDL LAB/Exp10_Autoencoders/autoencoder_denoising.py"
```

Quick smoke test:

```powershell
& "c:/Users/harsh/MLDL LAB/.venv/Scripts/python.exe" "c:/Users/harsh/MLDL LAB/Exp10_Autoencoders/autoencoder_denoising.py" --epochs 2 --train-subset 5000 --test-subset 1000
```

## Useful options

- `--epochs` (default: 20)
- `--batch-size` (default: 128)
- `--noise-factor` (default: 0.4)
- `--train-subset` (default: 0 = full train set)
- `--test-subset` (default: 0 = full test set)

Example full run with custom noise:

```powershell
& "c:/Users/harsh/MLDL LAB/.venv/Scripts/python.exe" "c:/Users/harsh/MLDL LAB/Exp10_Autoencoders/autoencoder_denoising.py" --epochs 15 --noise-factor 0.35
```

## Notes

- Quick runs may not outperform noisy baseline because of very short training.
- For better denoising quality, use full dataset and more epochs.
