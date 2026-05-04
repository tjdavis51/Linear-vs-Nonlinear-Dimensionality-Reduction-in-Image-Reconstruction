# Diffusion Bridge

This file places diffusion in the project without forcing it into the same
latent-compression table as PCA and AE.

## Why This Is Different

- PCA and AE reconstruct an input after compressing it into a fixed low-dimensional code.
- Diffusion does not do that. Instead, it learns to reverse progressive noise corruption.
- The bridge metric here is therefore denoising reconstruction quality: estimate `x0` from a noisy `xt` and score that estimate with PSNR/SSIM.

## Bridge Result

- dataset: MNIST
- train subset: 2048
- test subset: 512
- epochs: 1
- denoising PSNR: 18.931 dB
- denoising SSIM: 0.5365
- denoising MSE: 0.012791

## Interpretation

- This metric is useful for recognizing diffusion as an image-restoration/generative model rather than a dimensionality-reduction model.
- If you care about recovering structure from corruption and eventually generating plausible new samples, diffusion is worth doing.
- If you care about compact latent compression and direct reconstruction tradeoffs, PCA and AE remain the proper comparison set.
- So diffusion belongs in the project as a qualitative and denoising-oriented extension, not as the winner or loser of the latent-dimension benchmark.
