from __future__ import annotations

import math
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
AE_METRICS_PATH = ROOT.parent / "autoencoders" / "results" / "metrics.csv"
PCA_METRICS_PATH = ROOT / "results" / "full_split_pca" / "metrics" / "unified_metrics.csv"
EXPLORATORY_METRICS_PATH = ROOT / "results" / "metrics" / "unified_metrics.csv"
DIFFUSION_BRIDGE_PATH = (
    ROOT / "results" / "final_comparison" / "diffusion_bridge" / "diffusion_bridge_metrics.json"
)
OUTPUT_DIR = ROOT / "results" / "final_comparison"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ae_metrics() -> pd.DataFrame:
    df = pd.read_csv(AE_METRICS_PATH)
    dataset_map = {"mnist": "mnist", "fashion_mnist": "fashion"}
    df["dataset"] = df["dataset"].map(dataset_map)
    df["method"] = "ae"
    df["parameter_count"] = df["model_parameters"]
    df["compression_ratio"] = 784.0 / df["latent_dim"]
    df["mse"] = df["test_mse"]
    df["rmse"] = df["mse"].map(math.sqrt)
    df["mae"] = pd.NA
    df["psnr"] = df["test_psnr"]
    df["ssim"] = df["test_ssim"]
    df["mse_255"] = df["mse"] * (255.0**2)
    df["rmse_255"] = df["rmse"] * 255.0
    df["mae_255"] = pd.NA
    df["split"] = "standard_train_test"
    df["notes"] = "Committed AE result regenerated from the project autoencoder script."
    return df[
        [
            "dataset",
            "method",
            "latent_dim",
            "split",
            "epochs",
            "batch_size",
            "learning_rate",
            "parameter_count",
            "compression_ratio",
            "mse",
            "rmse",
            "mae",
            "psnr",
            "ssim",
            "mse_255",
            "rmse_255",
            "mae_255",
            "notes",
        ]
    ].copy()


def load_pca_metrics() -> pd.DataFrame:
    df = pd.read_csv(PCA_METRICS_PATH)
    df["method"] = "pca"
    df["split"] = "standard_train_test"
    df["notes"] = "Full-split PCA benchmark produced by comparison/run_unified_benchmark.py."
    return df[
        [
            "dataset",
            "method",
            "latent_dim",
            "split",
            "epochs",
            "batch_size",
            "learning_rate",
            "parameter_count",
            "compression_ratio",
            "mse",
            "rmse",
            "mae",
            "psnr",
            "ssim",
            "mse_255",
            "rmse_255",
            "mae_255",
            "notes",
        ]
    ].copy()


def build_core_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ae_df = load_ae_metrics()
    pca_df = load_pca_metrics()
    core_df = pd.concat([pca_df, ae_df], ignore_index=True).sort_values(
        ["dataset", "latent_dim", "method"]
    )

    ae_comp = ae_df.rename(
        columns={
            "mse": "ae_mse",
            "psnr": "ae_psnr",
            "ssim": "ae_ssim",
            "parameter_count": "ae_parameter_count",
            "rmse": "ae_rmse",
            "mse_255": "ae_mse_255",
            "rmse_255": "ae_rmse_255",
        }
    )
    pca_comp = pca_df.rename(
        columns={
            "mse": "pca_mse",
            "psnr": "pca_psnr",
            "ssim": "pca_ssim",
            "parameter_count": "pca_parameter_count",
            "rmse": "pca_rmse",
            "mae": "pca_mae",
            "mse_255": "pca_mse_255",
            "rmse_255": "pca_rmse_255",
            "mae_255": "pca_mae_255",
        }
    )
    joined = pca_comp.merge(
        ae_comp,
        on=["dataset", "latent_dim", "split", "compression_ratio"],
        how="inner",
    )
    joined["ae_minus_pca_psnr"] = joined["ae_psnr"] - joined["pca_psnr"]
    joined["ae_minus_pca_ssim"] = joined["ae_ssim"] - joined["pca_ssim"]
    joined["ae_minus_pca_mse"] = joined["ae_mse"] - joined["pca_mse"]
    joined["parameter_ratio_ae_to_pca"] = (
        joined["ae_parameter_count"] / joined["pca_parameter_count"]
    )
    joined["psnr_winner"] = joined["ae_minus_pca_psnr"].map(
        lambda delta: "ae" if delta > 0 else "pca"
    )
    joined["ssim_winner"] = joined["ae_minus_pca_ssim"].map(
        lambda delta: "ae" if delta > 0 else "pca"
    )
    joined["mse_winner"] = joined["ae_minus_pca_mse"].map(
        lambda delta: "ae" if delta < 0 else "pca"
    )
    return core_df, joined


def load_exploratory_notes() -> pd.DataFrame:
    df = pd.read_csv(EXPLORATORY_METRICS_PATH)
    return df[df["method"].isin(["dae", "vae"])].copy()


def load_diffusion_bridge() -> dict[str, object] | None:
    if not DIFFUSION_BRIDGE_PATH.is_file():
        return None
    return json.loads(DIFFUSION_BRIDGE_PATH.read_text(encoding="utf-8"))


def write_summary(
    core_df: pd.DataFrame,
    joined: pd.DataFrame,
    exploratory: pd.DataFrame,
    diffusion_bridge: dict[str, object] | None,
) -> None:
    lines = [
        "# Final Apples-to-Apples Comparison",
        "",
        "This summary uses one shared metric scale across models:",
        "",
        "- `MSE`, `RMSE`, `MAE` on normalized `0..1` pixels",
        "- `PSNR` in decibels, computed from normalized MSE",
        "- `SSIM` on normalized grayscale images",
        "",
        "The core table compares `PCA` and the committed `AE` results on the standard",
        "MNIST/Fashion-MNIST train/test split. This is the fairest direct comparison in",
        "the repository right now because both sides use the same latent sizes and full",
        "dataset protocol.",
        "",
        "The exploratory DAE/VAE benchmark is reported separately because it was run on a",
        "smaller 10k/2k split for speed and should not be mixed into the final headline",
        "table.",
        "",
    ]

    for dataset_name in ["mnist", "fashion"]:
        subset = joined[joined["dataset"] == dataset_name].sort_values("latent_dim")
        lines.append(f"## {dataset_name.upper()}")
        lines.append("")

        ae_psnr_wins = subset[subset["ae_minus_pca_psnr"] > 0]
        pca_psnr_wins = subset[subset["ae_minus_pca_psnr"] <= 0]

        if not ae_psnr_wins.empty:
            winning_dims = ", ".join(str(int(value)) for value in ae_psnr_wins["latent_dim"])
            lines.append(f"- AE has higher PSNR at latent dimensions: `{winning_dims}`.")
        if not pca_psnr_wins.empty:
            winning_dims = ", ".join(str(int(value)) for value in pca_psnr_wins["latent_dim"])
            lines.append(f"- PCA has higher PSNR at latent dimensions: `{winning_dims}`.")

        best_psnr_row = subset.sort_values(["ae_minus_pca_psnr"], ascending=False).iloc[0]
        biggest_pca_row = subset.sort_values(["ae_minus_pca_psnr"]).iloc[0]
        lines.append(
            f"- Largest AE advantage: latent `{int(best_psnr_row['latent_dim'])}`, "
            f"`AE {best_psnr_row['ae_psnr']:.3f} dB` vs `PCA {best_psnr_row['pca_psnr']:.3f} dB` "
            f"(`+{best_psnr_row['ae_minus_pca_psnr']:.3f} dB`)."
        )
        lines.append(
            f"- Largest PCA advantage: latent `{int(biggest_pca_row['latent_dim'])}`, "
            f"`AE {biggest_pca_row['ae_psnr']:.3f} dB` vs `PCA {biggest_pca_row['pca_psnr']:.3f} dB` "
            f"(`{biggest_pca_row['ae_minus_pca_psnr']:.3f} dB`)."
        )
        lines.append(
            f"- AE uses roughly `{subset['parameter_ratio_ae_to_pca'].min():.1f}x` to "
            f"`{subset['parameter_ratio_ae_to_pca'].max():.1f}x` more stored parameters "
            f"than PCA at the same latent size."
        )
        lines.append("")
        lines.append("```text")
        lines.append(
            subset[
                [
                    "latent_dim",
                    "pca_psnr",
                    "ae_psnr",
                    "ae_minus_pca_psnr",
                    "pca_ssim",
                    "ae_ssim",
                    "ae_minus_pca_ssim",
                    "pca_parameter_count",
                    "ae_parameter_count",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Interpretive Conclusions",
            "",
            "- On these simple 28x28 grayscale datasets, a fully connected AE can beat PCA at low-to-mid latent sizes, but PCA catches up or overtakes again once the latent space is large enough.",
            "- MNIST shows the clearest AE advantage in the middle of the curve: AE beats PCA at `8`, `16`, and `32` dimensions, but PCA retakes the lead at `64` dimensions.",
            "- Fashion-MNIST shows the same general pattern: AE is better at `8`, `16`, and slightly at `32`, while PCA becomes better again at `64`.",
            "- PCA is dramatically more parameter-efficient. At the same latent size it stores far fewer parameters and needs no gradient-based training, which makes it a very strong baseline when simplicity and efficiency matter.",
            "- The current nonlinear advantage is therefore conditional, not universal: use the AE when the latent bottleneck is small and reconstruction quality matters more than model size; use PCA when you want a fast, stable, interpretable baseline or when a moderate latent size already gives enough quality.",
            "",
            "## DAE and VAE Notes",
            "",
            "- The exploratory subset benchmark in `comparison/results/metrics/unified_metrics.csv` did not show DAE or VAE beating the core PCA/AE comparison on raw reconstruction quality.",
            "- That does not mean DAE or VAE are useless. DAE is usually justified by robustness to noisy inputs, and VAE is usually justified by latent-space structure and generation, not by best possible reconstruction MSE on easy grayscale datasets.",
            "- Diffusion should remain an extra-credit qualitative extension rather than part of the dimensionality reduction comparison because it is not compressing inputs into a fixed latent code for reconstruction.",
            "",
        ]
    )

    if diffusion_bridge is not None:
        lines.extend(
            [
                "## Diffusion Bridge",
                "",
                "- Diffusion is included through one denoising-reconstruction bridge metric instead of the latent-dimension table.",
                f"- Lightweight MNIST bridge result: `PSNR {float(diffusion_bridge['psnr']):.3f} dB`, `SSIM {float(diffusion_bridge['ssim']):.4f}`, `MSE {float(diffusion_bridge['mse']):.6f}`.",
                "- This score comes from estimating a clean image `x0` from a noisy intermediate `xt`, not from reconstructing after latent compression.",
                "- That means diffusion belongs in the project as a restoration/generative model: worth doing when denoising or generation is the goal, but not the right winner/loser comparison for dimensionality reduction.",
                "",
            ]
        )

    (OUTPUT_DIR / "final_conclusions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dir()
    core_df, joined = build_core_tables()
    exploratory = load_exploratory_notes()
    diffusion_bridge = load_diffusion_bridge()

    core_df.to_csv(OUTPUT_DIR / "core_metrics.csv", index=False)
    joined.to_csv(OUTPUT_DIR / "pca_vs_ae_comparison.csv", index=False)
    exploratory.to_csv(OUTPUT_DIR / "exploratory_dae_vae_metrics.csv", index=False)
    write_summary(core_df, joined, exploratory, diffusion_bridge)

    print("Saved", OUTPUT_DIR / "core_metrics.csv")
    print("Saved", OUTPUT_DIR / "pca_vs_ae_comparison.csv")
    print("Saved", OUTPUT_DIR / "exploratory_dae_vae_metrics.csv")
    print("Saved", OUTPUT_DIR / "final_conclusions.md")


if __name__ == "__main__":
    main()
