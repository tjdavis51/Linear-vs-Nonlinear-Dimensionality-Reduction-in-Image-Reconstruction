# Final Apples-to-Apples Comparison

This summary uses one shared metric scale across models:

- `MSE`, `RMSE`, `MAE` on normalized `0..1` pixels
- `PSNR` in decibels, computed from normalized MSE
- `SSIM` on normalized grayscale images

The core table compares `PCA` and the committed `AE` results on the standard
MNIST/Fashion-MNIST train/test split. This is the fairest direct comparison in
the repository right now because both sides use the same latent sizes and full
dataset protocol.

The exploratory DAE/VAE benchmark is reported separately because it was run on a
smaller 10k/2k split for speed and should not be mixed into the final headline
table.

## MNIST

- AE has higher PSNR at latent dimensions: `2, 8, 16, 32`.
- PCA has higher PSNR at latent dimensions: `64`.
- Largest AE advantage: latent `16`, `AE 18.757 dB` vs `PCA 15.955 dB` (`+2.801 dB`).
- Largest PCA advantage: latent `64`, `AE 19.394 dB` vs `PCA 21.105 dB` (`-1.711 dB`).
- AE uses roughly `9.5x` to `199.4x` more stored parameters than PCA at the same latent size.

```text
 latent_dim  pca_psnr  ae_psnr  ae_minus_pca_psnr  pca_ssim  ae_ssim  ae_minus_pca_ssim  pca_parameter_count  ae_parameter_count
          2   12.5591  13.9235             1.3644    0.5696   0.7146             0.1450                 2352              469010
          8   14.3876  17.0280             2.6404    0.7340   0.8754             0.1414                 7056              470552
         16   15.9551  18.7565             2.8014    0.8235   0.9190             0.0955                13328              472608
         32   18.1796  19.4316             1.2520    0.8972   0.9321             0.0350                25872              476720
         64   21.1054  19.3943            -1.7111    0.9486   0.9311            -0.0175                50960              484944
```

## FASHION

- AE has higher PSNR at latent dimensions: `2, 8, 16, 32`.
- PCA has higher PSNR at latent dimensions: `64`.
- Largest AE advantage: latent `8`, `AE 18.041 dB` vs `PCA 15.859 dB` (`+2.182 dB`).
- Largest PCA advantage: latent `64`, `AE 18.578 dB` vs `PCA 20.070 dB` (`-1.492 dB`).
- AE uses roughly `9.5x` to `199.4x` more stored parameters than PCA at the same latent size.

```text
 latent_dim  pca_psnr  ae_psnr  ae_minus_pca_psnr  pca_ssim  ae_ssim  ae_minus_pca_ssim  pca_parameter_count  ae_parameter_count
          2   13.4061  15.5102             2.1041    0.6896   0.8159             0.1263                 2352              469010
          8   15.8585  18.0410             2.1824    0.8240   0.8927             0.0687                 7056              470552
         16   17.0285  18.3954             1.3669    0.8658   0.9004             0.0347                13328              472608
         32   18.3878  18.5247             0.1369    0.9026   0.9031             0.0005                25872              476720
         64   20.0702  18.5781            -1.4920    0.9347   0.9041            -0.0307                50960              484944
```

## Interpretive Conclusions

- On these simple 28x28 grayscale datasets, a fully connected AE can beat PCA at low-to-mid latent sizes, but PCA catches up or overtakes again once the latent space is large enough.
- MNIST shows the clearest AE advantage in the middle of the curve: AE beats PCA at `8`, `16`, and `32` dimensions, but PCA retakes the lead at `64` dimensions.
- Fashion-MNIST shows the same general pattern: AE is better at `8`, `16`, and slightly at `32`, while PCA becomes better again at `64`.
- PCA is dramatically more parameter-efficient. At the same latent size it stores far fewer parameters and needs no gradient-based training, which makes it a very strong baseline when simplicity and efficiency matter.
- The current nonlinear advantage is therefore conditional, not universal: use the AE when the latent bottleneck is small and reconstruction quality matters more than model size; use PCA when you want a fast, stable, interpretable baseline or when a moderate latent size already gives enough quality.

## DAE and VAE Notes

- The exploratory subset benchmark in `comparison/results/metrics/unified_metrics.csv` did not show DAE or VAE beating the core PCA/AE comparison on raw reconstruction quality.
- That does not mean DAE or VAE are useless. DAE is usually justified by robustness to noisy inputs, and VAE is usually justified by latent-space structure and generation, not by best possible reconstruction MSE on easy grayscale datasets.
- Diffusion should remain an extra-credit qualitative extension rather than part of the dimensionality reduction comparison because it is not compressing inputs into a fixed latent code for reconstruction.

## Diffusion Bridge

- Diffusion is included through one denoising-reconstruction bridge metric instead of the latent-dimension table.
- Lightweight MNIST bridge result: `PSNR 18.931 dB`, `SSIM 0.5365`, `MSE 0.012791`.
- This score comes from estimating a clean image `x0` from a noisy intermediate `xt`, not from reconstructing after latent compression.
- That means diffusion belongs in the project as a restoration/generative model: worth doing when denoising or generation is the goal, but not the right winner/loser comparison for dimensionality reduction.
