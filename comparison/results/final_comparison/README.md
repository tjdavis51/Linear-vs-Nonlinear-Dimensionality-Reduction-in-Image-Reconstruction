# Final Comparison Package

This folder is the cleaned, commit-ready output package for the final project.

## Recommended Slide Assets

- `visuals/pca_vs_ae_metric_panels.png`
  Best single overview of the shared quantitative comparison.
- `visuals/parameter_efficiency_overlay.png`
  Shows the reconstruction-quality versus model-cost tradeoff.
- `visuals/sample_comparisons/fashion_side_by_side_summary.png`
  Best visual for the harder dataset and the most intuitive side-by-side story.
- `diffusion_bridge/diffusion_psnr_context.png`
  Best visual for placing diffusion in the project without forcing it into the latent-compression benchmark.

## Core Tables

- `core_metrics.csv`
  Combined PCA and AE metrics on the shared metric scale.
- `pca_vs_ae_comparison.csv`
  Direct PCA-versus-AE deltas and parameter-ratio comparisons.
- `final_conclusions.md`
  Final written interpretation of the whole project, including the diffusion bridge.

## Supporting Material

- `exploratory_dae_vae_metrics.csv`
  Smaller-split exploratory results for DAE and VAE.
- `visuals/sample_comparisons/`
  Same-sample PCA-versus-AE reconstruction grids.
- `diffusion_bridge/`
  Denoising-oriented diffusion context outputs.
