# Image Reconstruction

CS 472 final project comparing linear and nonlinear dimensionality reduction
for image reconstruction.

## Repository Layout

```text
pca/                     PCA experiment code and saved PCA artifacts
autoencoders/            AE, DAE, VAE code, regeneration script, and AE results
diffusion/               Extra-credit diffusion code, configs, and sample outputs
slurm/                   Cluster job helpers for the diffusion study
train.py                 Shared training entry point for AE/DAE/VAE/diffusion
requirements.txt         Shared Python dependencies
```

Each experiment area now owns its own committed result artifacts:

```text
pca/results/
autoencoders/results/
diffusion/results/
```

Generated training runs are not committed. By default:

- `python train.py --model ae ...` writes to `autoencoders/outputs/`
- `python train.py --model dae ...` writes to `autoencoders/outputs/`
- `python train.py --model vae ...` writes to `autoencoders/outputs/`
- `python train.py --model diffusion ...` writes to `diffusion/outputs/`
- multi-model sweeps fall back to `runs/`

## PCA

The linear baseline lives under `pca/`.

Run:

```bash
python pca/pca_mnist.py
```

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

Regenerate the saved report images:

```bash
python autoencoders/scripts/regenerate_autoencoder_images.py --dataset both --latent-dims 2 8 16 32 64 --epochs 10
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

Cluster job helpers remain under `slurm/final_study/` and now point at
`diffusion/configs/...`.
