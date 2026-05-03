# Image Reconstruction

CS 472 unsupervised learning final project comparing PCA and autoencoder-based image reconstruction on MNIST and Fashion-MNIST. PCA is teammate-owned work and final PCA figures/analysis will be merged later. This side of the repo contains AE, DAE, VAE, VAE interpolation/generation, and an extra-credit diffusion extension.

## Project Layout

```text
autoencoders/              AE, DAE, VAE models, training, metrics, plots
diffusion/                 Extra-credit DDPM code for MNIST and CIFAR-10
configs/diffusion/         mnist.yaml and cifar10.yaml diffusion configs
scripts/regenerate_autoencoder_images.py
results/autoencoders/      Final regenerated AE reconstruction PNGs
results/diffusion/         Final diffusion sample PNGs
slurm/final_study/         Reproducible diffusion smoke/MNIST/CIFAR jobs
train.py                   Main training entry point
requirements.txt           Python dependencies
```

Generated folders such as `data/`, `outputs/`, `runs/`, `checkpoints/`, and `deliverables/` are ignored. Final project PNGs live under `results/`.

## Final Result Files

The `results/` folder contains the small final artifacts intended for the project report and teammate comparison work. It does not contain datasets, model checkpoints, or training caches.

Autoencoder final outputs are under:

```text
results/autoencoders/
results/autoencoders/mnist/
results/autoencoders/fashion_mnist/
```

Each autoencoder dataset folder contains:

```text
latent_2.png
latent_8.png
latent_16.png
latent_32.png
latent_64.png
latent_comparison_grid.png
metrics.csv
metrics.json
```

The `latent_*.png` files show 10 fixed test images with originals on the first row and reconstructions on the second row. The `latent_comparison_grid.png` file shows the same fixed samples side-by-side across latent dimensions. These are the visual outputs to use directly in the report.

The root autoencoder metrics files combine both datasets:

```text
results/autoencoders/metrics.csv
results/autoencoders/metrics.json
```

Use the CSV files for plotting PCA-vs-AE comparisons. Each row is one dataset and latent dimension. Columns include `dataset`, `latent_dim`, `final_train_loss`, `test_mse`, `test_psnr`, `test_ssim`, `model_parameters`, `epochs`, `batch_size`, `learning_rate`, `seed`, and `device`.

Diffusion final sample images are under:

```text
results/diffusion/
```

These diffusion PNGs were moved into `results/diffusion/` and preserved as final result images. The diffusion code and generated training outputs were not changed.

## Autoencoder Runs

Regular autoencoder:

```bash
python train.py --model ae --dataset mnist --latent-dim 16
```

Denoising autoencoder:

```bash
python train.py --model dae --dataset mnist --latent-dim 16 --dae-noise-level 0.2
```

Variational autoencoder:

```bash
python train.py --model vae --dataset mnist --latent-dim 16
```

Run all three learned reconstruction models:

```bash
python train.py --models ae dae vae --dataset mnist --latent-dim 16
```

Fashion-MNIST is supported for the autoencoder side:

```bash
python train.py --model ae --dataset fashion --latent-dim 16
```

Autoencoder outputs are written under:

```text
outputs/<dataset>/<model>/<run_name>/
```

## Regenerating Autoencoder Results

Regenerate the final MNIST and Fashion-MNIST fully connected autoencoder reconstruction images:

```bash
python scripts/regenerate_autoencoder_images.py --dataset both --latent-dims 2 8 16 32 64 --epochs 10
```

The regeneration script trains or loads the fully connected autoencoder for each requested latent dimension, evaluates it on the deterministic test split, and writes both images and metrics. It uses the same fixed test samples for every latent dimension so the grids compare reconstruction quality cleanly.

Default output locations:

```text
results/autoencoders/mnist/
results/autoencoders/fashion_mnist/
results/autoencoders/metrics.csv
results/autoencoders/metrics.json
```

Useful options:

```bash
python scripts/regenerate_autoencoder_images.py --dataset mnist --latent-dims 2 8 16 32 64 --epochs 10
python scripts/regenerate_autoencoder_images.py --dataset fashion_mnist --latent-dims 2 8 16 32 64 --epochs 10
python scripts/regenerate_autoencoder_images.py --dataset both --latent-dims 2 8 16 32 64 --epochs 10 --device cpu
```

The script may download MNIST/Fashion-MNIST into `data/` if missing. `data/`, checkpoints, and model weights are ignored by git; only final PNG/CSV/JSON results are kept.

## Diffusion Extra Credit

Diffusion is an extra-credit extension, not the main PCA vs autoencoder comparison. The active diffusion workflow is intentionally limited to:

- MNIST: native `28x28`, grayscale, legacy U-Net config
- CIFAR-10: native `32x32`, RGB, ADM-style U-Net config

Fashion-MNIST diffusion and old checkpoint-only FID/LPIPS/Inception evaluation paths are not part of the cleaned workflow. A diffusion run trains the model and saves checkpoints, metrics, a loss curve, reconstructions, reverse-process snapshots, native sample grids, and generated samples.

Run from configs:

```bash
python train.py --config configs/diffusion/mnist.yaml
python train.py --config configs/diffusion/cifar10.yaml
```

Diffusion outputs are written under:

```text
outputs/<dataset>/diffusion/<run_name>/
```

Expected files include:

```text
checkpoints/best.pt
metrics.json
plots/loss_curve.png
plots/reconstructions.png
plots/diffusion_snapshots.png
samples/generated_samples.png
samples/generated_samples_native_grid.png
```

## Slurm Diffusion Runs

The small final-study Slurm surface is under `slurm/final_study/`:

```bash
sbatch slurm/final_study/smoke_array.slurm
sbatch slurm/final_study/train_mnist_array.slurm
sbatch slurm/final_study/train_cifar10_array.slurm
```

By default these write to:

```text
/scratch/$USER/image-reconstruction-final-study/runs/<dataset>/diffusion/<run_name>/
```

## PCA Status

PCA results are intentionally placeholders for now. The final report still needs teammate PCA reconstruction grids, PCA metrics, and the final PCA vs AE conclusion section.
