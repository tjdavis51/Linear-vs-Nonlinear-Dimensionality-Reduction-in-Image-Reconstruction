# Unified Reconstruction Comparison

This benchmark compares PCA, AE, DAE, and VAE on the same dataset split,
using the same reconstruction metrics on normalized `0..1` pixels.

Diffusion is excluded from the core table because it is not a latent
dimensionality reduction model and therefore is not an apples-to-apples
compression/reconstruction baseline against PCA or autoencoders.

## Benchmark Setup

- datasets: mnist, fashion
- methods: pca
- latent dims: 2, 8, 16, 32, 64
- train subset size: 60000
- test subset size: 10000
- epochs for learned models: 5
- batch size: 256

## MNIST

- best PSNR: `pca` at latent `64` with `21.105 dB`
- best SSIM: `pca` at latent `64` with `0.9486`
- strongest low-dimension point in this run: `pca` at latent `2` with `12.559 dB`

```text
method  latent_dim    mse    mae    psnr   ssim  parameter_count  train_seconds
   pca           2 0.0555 0.1286 12.5591 0.5696             2352         2.9916
   pca           8 0.0364 0.0968 14.3876 0.7340             7056         3.0132
   pca          16 0.0254 0.0770 15.9551 0.8235            13328         3.2056
   pca          32 0.0152 0.0569 18.1796 0.8972            25872         3.1153
   pca          64 0.0078 0.0388 21.1054 0.9486            50960         3.4360
```

## FASHION

- best PSNR: `pca` at latent `64` with `20.070 dB`
- best SSIM: `pca` at latent `64` with `0.9347`
- strongest low-dimension point in this run: `pca` at latent `2` with `13.406 dB`

```text
method  latent_dim    mse    mae    psnr   ssim  parameter_count  train_seconds
   pca           2 0.0456 0.1458 13.4061 0.6896             2352         3.4740
   pca           8 0.0260 0.0996 15.8585 0.8240             7056         3.4665
   pca          16 0.0198 0.0850 17.0285 0.8658            13328         3.7068
   pca          32 0.0145 0.0709 18.3878 0.9026            25872         3.9302
   pca          64 0.0098 0.0563 20.0702 0.9347            50960         4.2283
```
