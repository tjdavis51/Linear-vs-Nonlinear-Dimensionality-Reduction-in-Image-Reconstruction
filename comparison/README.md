# Unified Comparison

This folder contains a shared evaluation path for the dimensionality reduction
models that can be compared directly on reconstruction:

- `PCA`
- `AE`
- `DAE`
- `VAE`

Diffusion is intentionally not part of the core benchmark because it is not an
encoder-based dimensionality reduction model and does not compress an input
image into a fixed low-dimensional latent code in the same way as PCA or the
autoencoder family.

Instead, diffusion is tied in through a separate bridge analysis that scores
denoising-based image recovery with PSNR and SSIM. That keeps diffusion in the
project honestly as a restoration/generative model instead of forcing it into a
latent-compression leaderboard.

Run the benchmark with:

```bash
python comparison/run_unified_benchmark.py --download
```

Outputs are written under:

```text
comparison/results/
```

Final presentation-ready materials are under:

```text
comparison/results/final_comparison/
```
