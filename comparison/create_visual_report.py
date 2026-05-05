from __future__ import annotations

import math
import os
import random
import sys
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
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.optim import Adam
from torch.utils.data import DataLoader

from autoencoders.data import build_autoencoder_dataset
from autoencoders.models import FullyConnectedAutoencoder
from autoencoders.training import train_epoch


ROOT = Path(__file__).resolve().parent
FINAL_COMPARISON_DIR = ROOT / "results" / "final_comparison"
VISUAL_DIR = FINAL_COMPARISON_DIR / "visuals"
SAMPLE_DIR = VISUAL_DIR / "sample_comparisons"
PREPROCESS_DIR = VISUAL_DIR / "preprocessing"
CHECKPOINT_DIR = ROOT / "checkpoints"
CORE_METRICS_PATH = FINAL_COMPARISON_DIR / "core_metrics.csv"
JOINED_METRICS_PATH = FINAL_COMPARISON_DIR / "pca_vs_ae_comparison.csv"

LATENT_DIMS_FOR_SAMPLES = (16, 64)
SAMPLE_COUNT = 8
PREPROCESS_SAMPLE_COUNT = 5
SEED = 42


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


def ensure_dirs() -> None:
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    core = pd.read_csv(CORE_METRICS_PATH)
    joined = pd.read_csv(JOINED_METRICS_PATH)
    return core, joined


def plot_metric_panels(core: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    datasets = [("mnist", "MNIST"), ("fashion", "Fashion-MNIST")]
    metrics = [("psnr", "PSNR (dB)"), ("ssim", "SSIM")]

    for row, (dataset_key, dataset_label) in enumerate(datasets):
        subset = core[core["dataset"] == dataset_key]
        for col, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for method, color in (("pca", "#1f77b4"), ("ae", "#d62728")):
                method_df = subset[subset["method"] == method].sort_values("latent_dim")
                ax.plot(
                    method_df["latent_dim"],
                    method_df[metric_key],
                    marker="o",
                    linewidth=2,
                    label=method.upper(),
                    color=color,
                )
            ax.set_title(f"{dataset_label}: {metric_label}")
            ax.set_xlabel("Latent Dimension")
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.25)
            if row == 0 and col == 1:
                ax.legend()

    fig.suptitle("PCA vs AE on Shared Reconstruction Metrics", fontsize=16)
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "pca_vs_ae_metric_panels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_psnr_delta(joined: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (dataset_key, dataset_label) in zip(
        axes, (("mnist", "MNIST"), ("fashion", "Fashion-MNIST"))
    ):
        subset = joined[joined["dataset"] == dataset_key].sort_values("latent_dim")
        values = subset["ae_minus_pca_psnr"].to_numpy()
        colors = ["#2ca02c" if value > 0 else "#9467bd" for value in values]
        ax.bar(subset["latent_dim"].astype(str), values, color=colors)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"{dataset_label}: AE minus PCA PSNR")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("PSNR Delta (dB)")
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "ae_minus_pca_psnr.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_efficiency(joined: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (dataset_key, dataset_label) in zip(
        axes, (("mnist", "MNIST"), ("fashion", "Fashion-MNIST"))
    ):
        subset = joined[joined["dataset"] == dataset_key].sort_values("latent_dim")
        ax.plot(
            subset["latent_dim"],
            subset["pca_psnr"],
            marker="o",
            linewidth=2,
            color="#1f77b4",
            label="PCA",
        )
        ax.plot(
            subset["latent_dim"],
            subset["ae_psnr"],
            marker="o",
            linewidth=2,
            color="#d62728",
            label="AE",
        )
        for _, row in subset.iterrows():
            ax.annotate(
                f"{row['parameter_ratio_ae_to_pca']:.0f}x",
                (row["latent_dim"], row["ae_psnr"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#444444",
            )
        ax.set_title(f"{dataset_label}: Quality vs Model Cost")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "parameter_efficiency_overlay.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_winner_heatmap(joined: pd.DataFrame) -> None:
    datasets = ["mnist", "fashion"]
    latents = sorted(joined["latent_dim"].unique())
    matrix = np.zeros((len(datasets), len(latents)), dtype=float)
    for row_idx, dataset_key in enumerate(datasets):
        subset = joined[joined["dataset"] == dataset_key].set_index("latent_dim")
        for col_idx, latent_dim in enumerate(latents):
            matrix[row_idx, col_idx] = subset.loc[latent_dim, "ae_minus_pca_psnr"]

    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(latents)), [str(dim) for dim in latents])
    ax.set_yticks(range(len(datasets)), ["MNIST", "Fashion-MNIST"])
    ax.set_xlabel("Latent Dimension")
    ax.set_title("AE Advantage Over PCA in PSNR (dB)")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    fig.colorbar(image, ax=ax, label="AE minus PCA PSNR (dB)")
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "psnr_delta_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def train_or_load_ae(
    dataset_name: str,
    latent_dim: int,
    device: torch.device,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> FullyConnectedAutoencoder:
    checkpoint_path = CHECKPOINT_DIR / f"{dataset_name}_ae_latent_{latent_dim}.pth"
    model = FullyConnectedAutoencoder(latent_dim=latent_dim).to(device)
    if checkpoint_path.is_file():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    train_dataset = build_autoencoder_dataset(
        dataset_name, root=REPO_ROOT / "data", train=True, download=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            model_type="ae",
            noise_level=0.0,
        )
    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    return model


def create_sample_grid(
    dataset_name: str,
    dataset_label: str,
    latent_dim: int,
    device: torch.device,
) -> None:
    train_dataset = build_autoencoder_dataset(
        dataset_name, root=REPO_ROOT / "data", train=True, download=True
    )
    test_dataset = build_autoencoder_dataset(
        dataset_name, root=REPO_ROOT / "data", train=False, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    train_images = []
    for batch, _ in train_loader:
        train_images.append(batch.view(batch.size(0), -1).numpy())
    x_train = np.concatenate(train_images, axis=0).astype(np.float32)

    sample_indices = list(range(SAMPLE_COUNT))
    original_batch = torch.stack([test_dataset[idx][0] for idx in sample_indices], dim=0)
    x_test = original_batch.view(original_batch.size(0), -1).numpy().astype(np.float32)

    pca = PCA(n_components=latent_dim, svd_solver="randomized", random_state=SEED)
    pca.fit(x_train)
    pca_recon = pca.inverse_transform(pca.transform(x_test)).astype(np.float32)
    pca_recon = np.clip(pca_recon, 0.0, 1.0)
    pca_tensor = torch.from_numpy(pca_recon).view(-1, 1, 28, 28)

    ae_model = train_or_load_ae(dataset_name, latent_dim, device)
    with torch.no_grad():
        ae_tensor = ae_model(original_batch.to(device)).detach().cpu().clamp(0.0, 1.0)

    fig, axes = plt.subplots(3, SAMPLE_COUNT, figsize=(1.8 * SAMPLE_COUNT, 5.4))
    row_labels = ("Original", "PCA", "AE")
    row_tensors = (original_batch, pca_tensor, ae_tensor)

    for row_idx, (label, tensor) in enumerate(zip(row_labels, row_tensors)):
        for col_idx in range(SAMPLE_COUNT):
            axes[row_idx, col_idx].imshow(
                tensor[col_idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"Sample {col_idx + 1}", fontsize=9)
        axes[row_idx, 0].set_ylabel(label, rotation=0, labelpad=36, va="center", fontsize=10)

    fig.suptitle(
        f"{dataset_label}: Original vs PCA vs AE (latent dim = {latent_dim})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        SAMPLE_DIR / f"{dataset_name}_pca_vs_ae_latent_{latent_dim:03d}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_fashion_summary_strip(device: torch.device) -> None:
    dataset_name = "fashion"
    dataset_label = "Fashion-MNIST"
    train_dataset = build_autoencoder_dataset(
        dataset_name, root=REPO_ROOT / "data", train=True, download=True
    )
    test_dataset = build_autoencoder_dataset(
        dataset_name, root=REPO_ROOT / "data", train=False, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    train_images = []
    for batch, _ in train_loader:
        train_images.append(batch.view(batch.size(0), -1).numpy())
    x_train = np.concatenate(train_images, axis=0).astype(np.float32)

    sample_indices = list(range(4))
    originals = torch.stack([test_dataset[idx][0] for idx in sample_indices], dim=0)
    x_test = originals.view(originals.size(0), -1).numpy().astype(np.float32)

    fig, axes = plt.subplots(
        1 + len(LATENT_DIMS_FOR_SAMPLES) * 2,
        len(sample_indices),
        figsize=(1.8 * len(sample_indices), 1.7 * (1 + len(LATENT_DIMS_FOR_SAMPLES) * 2)),
    )

    for col_idx in range(len(sample_indices)):
        axes[0, col_idx].imshow(originals[col_idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col_idx].axis("off")
        axes[0, col_idx].set_title(f"Sample {col_idx + 1}", fontsize=9)
    axes[0, 0].set_ylabel("Original", rotation=0, labelpad=28, va="center", fontsize=9)

    for offset, latent_dim in enumerate(LATENT_DIMS_FOR_SAMPLES):
        pca = PCA(n_components=latent_dim, svd_solver="randomized", random_state=SEED)
        pca.fit(x_train)
        pca_recon = pca.inverse_transform(pca.transform(x_test)).astype(np.float32)
        pca_recon = np.clip(pca_recon, 0.0, 1.0)
        pca_tensor = torch.from_numpy(pca_recon).view(-1, 1, 28, 28)

        ae_model = train_or_load_ae(dataset_name, latent_dim, device)
        with torch.no_grad():
            ae_tensor = ae_model(originals.to(device)).detach().cpu().clamp(0.0, 1.0)

        pca_row = 1 + offset * 2
        ae_row = pca_row + 1
        for col_idx in range(len(sample_indices)):
            axes[pca_row, col_idx].imshow(
                pca_tensor[col_idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[pca_row, col_idx].axis("off")
            axes[ae_row, col_idx].imshow(
                ae_tensor[col_idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[ae_row, col_idx].axis("off")

        axes[pca_row, 0].set_ylabel(
            f"PCA z={latent_dim}", rotation=0, labelpad=30, va="center", fontsize=9
        )
        axes[ae_row, 0].set_ylabel(
            f"AE z={latent_dim}", rotation=0, labelpad=30, va="center", fontsize=9
        )

    fig.suptitle(f"{dataset_label}: Side-by-Side Reconstruction Quality", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        SAMPLE_DIR / "fashion_side_by_side_summary.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_preprocessing_sample_grids() -> None:
    datasets = (
        ("mnist", "MNIST"),
        ("fashion", "Fashion-MNIST"),
    )

    fig, axes = plt.subplots(
        len(datasets),
        PREPROCESS_SAMPLE_COUNT,
        figsize=(1.8 * PREPROCESS_SAMPLE_COUNT, 2.6 * len(datasets)),
        squeeze=False,
    )

    for row_idx, (dataset_name, dataset_label) in enumerate(datasets):
        dataset = build_autoencoder_dataset(
            dataset_name, root=REPO_ROOT / "data", train=False, download=True
        )
        for col_idx in range(PREPROCESS_SAMPLE_COUNT):
            image, label = dataset[col_idx]
            axes[row_idx, col_idx].imshow(
                image[0].numpy(), cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(f"Sample {col_idx + 1}", fontsize=9)
            if row_idx == 0:
                axes[row_idx, col_idx].text(
                    0.5,
                    -0.14,
                    f"digit {label}",
                    transform=axes[row_idx, col_idx].transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
            else:
                axes[row_idx, col_idx].text(
                    0.5,
                    -0.14,
                    f"class {label}",
                    transform=axes[row_idx, col_idx].transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )

        axes[row_idx, 0].set_ylabel(
            f"{dataset_label}\n28x28\nnormalized",
            rotation=0,
            labelpad=38,
            va="center",
            fontsize=9,
        )

    fig.suptitle("Datasets After Preprocessing", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        PREPROCESS_DIR / "mnist_fashion_preprocessing_samples.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    seed_everything(SEED)
    ensure_dirs()
    device = resolve_device()
    core, joined = load_tables()

    plot_metric_panels(core)
    plot_psnr_delta(joined)
    plot_parameter_efficiency(joined)
    plot_winner_heatmap(joined)
    create_preprocessing_sample_grids()

    create_sample_grid("mnist", "MNIST", 16, device)
    create_sample_grid("mnist", "MNIST", 64, device)
    create_sample_grid("fashion", "Fashion-MNIST", 16, device)
    create_sample_grid("fashion", "Fashion-MNIST", 64, device)
    create_fashion_summary_strip(device)

    print("Saved visual report assets to", VISUAL_DIR)


if __name__ == "__main__":
    main()
