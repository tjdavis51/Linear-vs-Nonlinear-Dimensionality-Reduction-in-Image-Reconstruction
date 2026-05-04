# Unified Reconstruction Comparison

This benchmark compares PCA, AE, DAE, and VAE on the same dataset split,
using the same reconstruction metrics on normalized `0..1` pixels.

Diffusion is excluded from the core table because it is not a latent
dimensionality reduction model and therefore is not an apples-to-apples
compression/reconstruction baseline against PCA or autoencoders.

## Benchmark Setup

- datasets: mnist, fashion
- methods: pca, ae, dae, vae
- latent dims: 8, 16, 32, 64
- train subset size: 10000
- test subset size: 2000
- epochs for learned models: 5
- batch size: 256

## MNIST

- best PSNR: `pca` at latent `64` with `20.992 dB`
- best SSIM: `pca` at latent `64` with `0.9476`
- strongest low-dimension point in this run: `vae` at latent `8` with `14.640 dB`

```text
method  latent_dim    mse    mae    psnr   ssim  parameter_count  train_seconds
    ae           8 0.0446 0.1056 13.5021 0.6760           470552         7.9833
   dae           8    NaN    NaN     NaN    NaN           470552         3.9701
   pca           8 0.0370 0.0978 14.3228 0.7318             7056         0.4702
   vae           8 0.0344 0.0897 14.6400 0.7565           638400         5.7887
    ae          16 0.0448 0.1043 13.4910 0.6728           472608         4.2835
   dae          16    NaN    NaN     NaN    NaN           472608         4.0154
   pca          16 0.0258 0.0778 15.8795 0.8213            13328         0.4986
   vae          16 0.0328 0.0878 14.8389 0.7702           648016         4.1790
    ae          32 0.0436 0.1007 13.6053 0.6905           476720         4.1545
   dae          32 0.0689 0.1671 11.6184 0.5154           476720         4.2644
   pca          32 0.0156 0.0578 18.0730 0.8954            25872         0.5297
   vae          32 0.0330 0.0906 14.8096 0.7716           667248         4.3179
    ae          64 0.0497 0.1116 13.0372 0.6193           484944         4.3828
   dae          64 0.0702 0.1788 11.5374 0.4754           484944         4.4482
   pca          64 0.0080 0.0395 20.9924 0.9476            50960         0.6327
   vae          64    NaN    NaN     NaN    NaN           705712         4.6572
```

## FASHION

- best PSNR: `pca` at latent `64` with `20.011 dB`
- best SSIM: `pca` at latent `64` with `0.9338`
- strongest low-dimension point in this run: `pca` at latent `8` with `15.797 dB`

```text
method  latent_dim    mse    mae    psnr   ssim  parameter_count  train_seconds
    ae           8    NaN    NaN     NaN    NaN           470552         4.2133
   dae           8    NaN    NaN     NaN    NaN           470552         4.5597
   pca           8 0.0263 0.1004 15.7973 0.8226             7056         0.5263
   vae           8    NaN    NaN     NaN    NaN           638400         4.6436
    ae          16 0.0317 0.1055 14.9947 0.7858           472608         4.8186
   dae          16 0.0349 0.1118 14.5699 0.7836           472608         4.6775
   pca          16 0.0200 0.0855 16.9889 0.8651            13328         0.5612
   vae          16    NaN    NaN     NaN    NaN           648016         4.7877
    ae          32 0.0289 0.1004 15.3936 0.8034           476720         4.7711
   dae          32 0.0344 0.1114 14.6328 0.7886           476720         4.8442
   pca          32 0.0146 0.0714 18.3421 0.9019            25872         0.6649
   vae          32 0.0293 0.1065 15.3266 0.8186           667248         4.9432
    ae          64 0.0305 0.1037 15.1574 0.7905           484944         4.8961
   dae          64 0.0358 0.1152 14.4635 0.7832           484944         5.4984
   pca          64 0.0100 0.0568 20.0109 0.9338            50960         0.7463
   vae          64 0.0332 0.1135 14.7908 0.8065           705712         4.9547
```
