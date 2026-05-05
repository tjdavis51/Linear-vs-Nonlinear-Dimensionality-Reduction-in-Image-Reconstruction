# Image Reconstruction

CS 472 final project comparing linear and nonlinear dimensionality reduction
for image reconstruction.

## Quick Start

Create a virtual environment and install the shared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

This project uses `MNIST` and `Fashion-MNIST`.

- The submission package includes the raw dataset files under `data/`.
- If `data/` is missing, the PyTorch dataset loaders will download the files
  automatically the first time the training or comparison scripts run.

Expected dataset layout:

```text
data/
  MNIST/
  FashionMNIST/
```

## Repository Layout

```text
pca/                     PCA experiment code and saved PCA artifacts
autoencoders/            AE, DAE, VAE code, regeneration script, and AE results
diffusion/               Extra-credit diffusion code, configs, and sample outputs
comparison/              Unified metrics, final visuals, and comparison scripts
slurm/                   Cluster job helpers for the diffusion study
train.py                 Shared training entry point for AE/DAE/VAE/diffusion
requirements.txt         Shared Python dependencies
```

Committed result artifacts live under:

```text
pca/results/
autoencoders/results/
diffusion/results/
comparison/results/final_comparison/
```

Generated training runs are not committed. By default:

- `python train.py --model ae ...` writes to `autoencoders/outputs/`
- `python train.py --model dae ...` writes to `autoencoders/outputs/`
- `python train.py --model vae ...` writes to `autoencoders/outputs/`
- `python train.py --model diffusion ...` writes to `diffusion/outputs/`
- multi-model sweeps fall back to `runs/`

## Reproduce Main Results

Run the PCA baseline:

```bash
python pca/pca_mnist.py
```

Train a representative autoencoder run:

```bash
python train.py --model ae --dataset mnist --latent-dim 16
```

Regenerate the report-ready AE figures:

```bash
python autoencoders/scripts/regenerate_autoencoder_images.py --dataset both --latent-dims 2 8 16 32 64 --epochs 10
```

Regenerate the unified comparison metrics and visuals:

```bash
python comparison/run_unified_benchmark.py
python comparison/create_visual_report.py
python comparison/assemble_final_comparison.py
```

## PCA

The linear baseline lives under `pca/`.

Outputs:

```text
pca/results/metrics/
pca/results/plots/
pca/results/reconstructions/
```

## Autoencoders

The learned reconstruction models live under `autoencoders/`.

Train:

```bash
python train.py --model ae --dataset mnist --latent-dim 16
python train.py --model dae --dataset mnist --latent-dim 16 --dae-noise-level 0.2
python train.py --model vae --dataset mnist --latent-dim 16
```

Committed artifacts:

```text
autoencoders/results/mnist/
autoencoders/results/fashion_mnist/
autoencoders/results/metrics.csv
autoencoders/results/metrics.json
```

## Diffusion

The extra-credit diffusion work lives under `diffusion/`.

Run:

```bash
python train.py --config diffusion/configs/mnist.yaml
python train.py --config diffusion/configs/cifar10.yaml
```

Committed artifacts:

```text
diffusion/results/
```

## Final Comparison Package

The final presentation-ready comparison package lives under:

```text
comparison/results/final_comparison/
```

Key deliverables:

- `core_metrics.csv`
- `pca_vs_ae_comparison.csv`
- `final_conclusions.md`
- `visuals/`
- `diffusion_bridge/`

Cluster job helpers remain under `slurm/final_study/` and point at
`diffusion/configs/...`.
