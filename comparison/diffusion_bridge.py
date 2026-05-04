from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".cache" / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from diffusion.artifacts import plot_diffusion_reconstructions
from diffusion.data import build_diffusion_dataset
from diffusion.model import DiffusionUNet
from diffusion.scheduler import get_noise_schedule
from diffusion.training import (
    eval_diffusion_epoch,
    evaluate_diffusion_metrics,
    train_diffusion_epoch,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "final_comparison" / "diffusion_bridge"
CORE_METRICS_PATH = ROOT / "results" / "final_comparison" / "core_metrics.csv"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device() -> torch.device:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if mps_available:
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_subset(dataset, limit: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:limit]
    return Subset(dataset, indices.tolist())


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_bridge_experiment() -> dict[str, float | int | str]:
    seed = 42
    seed_everything(seed)
    device = resolve_device()

    train_dataset = build_diffusion_dataset(
        "mnist",
        root=REPO_ROOT / "data",
        train=True,
        image_size=28,
        channels=1,
        preprocessing_protocol="default",
        download=True,
    )
    test_dataset = build_diffusion_dataset(
        "mnist",
        root=REPO_ROOT / "data",
        train=False,
        image_size=28,
        channels=1,
        preprocessing_protocol="default",
        download=True,
    )

    train_subset = make_subset(train_dataset, limit=2048, seed=seed)
    test_subset = make_subset(test_dataset, limit=512, seed=seed + 1)

    train_loader = DataLoader(
        train_subset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = DiffusionUNet(
        in_channels=1,
        base_channels=32,
        time_dim=64,
        num_res_blocks=1,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    schedule = get_noise_schedule(
        T=100,
        device=device,
        beta_start=1e-4,
        beta_end=2e-2,
        schedule_name="linear",
    )

    start = time.perf_counter()
    train_loss = train_diffusion_epoch(
        model,
        train_loader,
        optimizer,
        schedule,
        device,
        prediction_type="eps",
        amp_dtype="none",
        grad_clip_norm=1.0,
    )
    val_loss = eval_diffusion_epoch(
        model,
        test_loader,
        schedule,
        device,
        prediction_type="eps",
        amp_dtype="none",
    )
    metrics = evaluate_diffusion_metrics(
        model,
        test_loader,
        schedule,
        device,
        prediction_type="eps",
        amp_dtype="none",
    )
    elapsed = time.perf_counter() - start

    plot_diffusion_reconstructions(
        model,
        schedule,
        test_loader,
        device,
        dataset_name="mnist",
        base_channels=32,
        prediction_type="eps",
        save_path=OUTPUT_DIR / "diffusion_denoising_preview.png",
        num_images=8,
    )

    payload = {
        "dataset": "mnist",
        "task": "denoising_reconstruction_estimate",
        "train_subset": 2048,
        "test_subset": 512,
        "epochs": 1,
        "timesteps": 100,
        "base_channels": 32,
        "num_res_blocks": 1,
        "device": device.type,
        "train_loss": float(train_loss),
        "val_noise_mse": float(val_loss),
        "mse": float(metrics["mse"]),
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "elapsed_seconds": float(elapsed),
        "note": (
            "Diffusion score is a denoising-based x0 reconstruction estimate, not a latent "
            "compression reconstruction metric."
        ),
    }
    (OUTPUT_DIR / "diffusion_bridge_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def create_context_plot(bridge_metrics: dict[str, float | int | str]) -> None:
    core = pd.read_csv(CORE_METRICS_PATH)
    mnist = core[core["dataset"] == "mnist"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color in (("pca", "#1f77b4"), ("ae", "#d62728")):
        subset = mnist[mnist["method"] == method].sort_values("latent_dim")
        ax.plot(
            subset["latent_dim"],
            subset["psnr"],
            marker="o",
            linewidth=2,
            color=color,
            label=f"{method.upper()} reconstruction",
        )

    diffusion_psnr = float(bridge_metrics["psnr"])
    ax.axhline(
        diffusion_psnr,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label="Diffusion denoising PSNR",
    )
    ax.text(
        64.5,
        diffusion_psnr + 0.08,
        "different task:\nnoisy x0 estimate",
        color="#2ca02c",
        fontsize=9,
        ha="left",
        va="bottom",
    )

    ax.set_title("Where Diffusion Fits: Denoising PSNR Context on MNIST")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diffusion_psnr_context.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_bridge_summary(bridge_metrics: dict[str, float | int | str]) -> None:
    lines = [
        "# Diffusion Bridge",
        "",
        "This file places diffusion in the project without forcing it into the same",
        "latent-compression table as PCA and AE.",
        "",
        "## Why This Is Different",
        "",
        "- PCA and AE reconstruct an input after compressing it into a fixed low-dimensional code.",
        "- Diffusion does not do that. Instead, it learns to reverse progressive noise corruption.",
        "- The bridge metric here is therefore denoising reconstruction quality: estimate `x0` from a noisy `xt` and score that estimate with PSNR/SSIM.",
        "",
        "## Bridge Result",
        "",
        f"- dataset: MNIST",
        f"- train subset: {bridge_metrics['train_subset']}",
        f"- test subset: {bridge_metrics['test_subset']}",
        f"- epochs: {bridge_metrics['epochs']}",
        f"- denoising PSNR: {float(bridge_metrics['psnr']):.3f} dB",
        f"- denoising SSIM: {float(bridge_metrics['ssim']):.4f}",
        f"- denoising MSE: {float(bridge_metrics['mse']):.6f}",
        "",
        "## Interpretation",
        "",
        "- This metric is useful for recognizing diffusion as an image-restoration/generative model rather than a dimensionality-reduction model.",
        "- If you care about recovering structure from corruption and eventually generating plausible new samples, diffusion is worth doing.",
        "- If you care about compact latent compression and direct reconstruction tradeoffs, PCA and AE remain the proper comparison set.",
        "- So diffusion belongs in the project as a qualitative and denoising-oriented extension, not as the winner or loser of the latent-dimension benchmark.",
        "",
    ]
    (OUTPUT_DIR / "diffusion_bridge_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    ensure_output_dir()
    bridge_metrics = run_bridge_experiment()
    create_context_plot(bridge_metrics)
    write_bridge_summary(bridge_metrics)
    print("Saved diffusion bridge outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
