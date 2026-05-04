from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
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
from torch.utils.data import DataLoader, Dataset, Subset

from autoencoders.data import build_autoencoder_dataset
from autoencoders.models import FullyConnectedAutoencoder, VariationalAutoencoder
from autoencoders.training import (
    compute_batch_ssim,
    eval_epoch,
    evaluate_metrics,
    train_epoch,
)


INPUT_DIM = 28 * 28
DEFAULT_DATASETS = ("mnist", "fashion")
DEFAULT_METHODS = ("pca", "ae", "dae", "vae")
DEFAULT_LATENT_DIMS = (8, 16, 32, 64)
DEFAULT_GRID_DIMS = (16, 64)


@dataclass(frozen=True)
class BenchmarkConfig:
    datasets: tuple[str, ...]
    methods: tuple[str, ...]
    latent_dims: tuple[int, ...]
    train_limit: int
    test_limit: int
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    seed: int
    noise_level: float
    output_dir: Path
    data_dir: Path
    device: str
    download: bool


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Run a shared PCA/AE/DAE/VAE reconstruction benchmark."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("mnist", "fashion"),
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("pca", "ae", "dae", "vae"),
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--latent-dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_LATENT_DIMS),
    )
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("comparison/results")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    if any(dim < 1 for dim in args.latent_dims):
        parser.error("--latent-dims values must all be positive")
    if args.train_limit < 1 or args.test_limit < 1:
        parser.error("--train-limit and --test-limit must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    return BenchmarkConfig(
        datasets=tuple(args.datasets),
        methods=tuple(args.methods),
        latent_dims=tuple(args.latent_dims),
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        noise_level=args.noise_level,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        device=args.device,
        download=args.download,
    )


def resolve_device(requested: str) -> torch.device:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    paths = {
        "root": base_dir,
        "metrics": base_dir / "metrics",
        "plots": base_dir / "plots",
        "grids": base_dir / "grids",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def select_subset(dataset: Dataset, limit: int, seed: int) -> Subset:
    generator = np.random.default_rng(seed)
    indices = generator.permutation(len(dataset))[:limit]
    return Subset(dataset, indices.tolist())


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def dataset_to_numpy(dataset: Dataset, batch_size: int = 512) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batches = []
    for images, _ in loader:
        batches.append(images.view(images.size(0), -1).numpy())
    return np.concatenate(batches, axis=0).astype(np.float32)


def dataset_to_tensor(dataset: Dataset, batch_size: int = 512) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batches = []
    for images, _ in loader:
        batches.append(images)
    return torch.cat(batches, dim=0)


def compute_reconstruction_metrics(
    originals: torch.Tensor, reconstructions: torch.Tensor, batch_size: int = 256
) -> dict[str, float]:
    originals = originals.detach().cpu().float()
    reconstructions = reconstructions.detach().cpu().float().clamp(0.0, 1.0)

    squared_error = (reconstructions - originals).pow(2)
    absolute_error = (reconstructions - originals).abs()
    mse = float(squared_error.mean().item())
    rmse = float(math.sqrt(mse))
    mae = float(absolute_error.mean().item())
    psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-12)))

    ssim_weighted_total = 0.0
    total_samples = 0
    for start in range(0, originals.size(0), batch_size):
        end = min(start + batch_size, originals.size(0))
        batch_originals = originals[start:end]
        batch_recons = reconstructions[start:end]
        batch_ssim = compute_batch_ssim(batch_recons, batch_originals)
        batch_count = end - start
        ssim_weighted_total += batch_ssim * batch_count
        total_samples += batch_count

    ssim = ssim_weighted_total / max(total_samples, 1)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "ssim": float(ssim),
        "mse_255": mse * (255.0**2),
        "rmse_255": rmse * 255.0,
        "mae_255": mae * 255.0,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def pca_storage_parameters(latent_dim: int) -> int:
    return INPUT_DIM * latent_dim + INPUT_DIM


def save_method_grid(
    originals: torch.Tensor,
    reconstructions_by_method: dict[str, torch.Tensor],
    output_path: Path,
    title: str,
    sample_count: int = 8,
) -> None:
    methods = list(reconstructions_by_method)
    rows = 1 + len(methods)
    fig, axes = plt.subplots(rows, sample_count, figsize=(1.8 * sample_count, 2.0 * rows))
    if rows == 1:
        axes = np.array([axes])

    for col in range(sample_count):
        axes[0, col].imshow(originals[col, 0].cpu().numpy(), cmap="gray")
        axes[0, col].set_title(f"Orig {col + 1}", fontsize=8)
        axes[0, col].axis("off")

    for row, method in enumerate(methods, start=1):
        for col in range(sample_count):
            axes[row, col].imshow(
                reconstructions_by_method[method][col, 0].cpu().numpy(), cmap="gray"
            )
            if col == 0:
                axes[row, col].set_ylabel(method.upper(), rotation=90, fontsize=8)
            axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_latent_dim(
    metrics_df: pd.DataFrame,
    *,
    dataset_name: str,
    metric: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    dataset_df = metrics_df[metrics_df["dataset"] == dataset_name].copy()
    for method in sorted(dataset_df["method"].unique()):
        method_df = dataset_df[dataset_df["method"] == method].sort_values("latent_dim")
        ax.plot(method_df["latent_dim"], method_df[metric], marker="o", label=method.upper())
    ax.set_title(f"{dataset_name.upper()} {metric.upper()} by Latent Dimension")
    ax.set_xlabel("Latent Dimension / Components")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(
    metrics_df: pd.DataFrame, *, dataset_name: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    dataset_df = metrics_df[metrics_df["dataset"] == dataset_name].copy()
    for method in sorted(dataset_df["method"].unique()):
        method_df = dataset_df[dataset_df["method"] == method]
        ax.scatter(
            method_df["parameter_count"],
            method_df["psnr"],
            label=method.upper(),
            s=60,
        )
        for _, row in method_df.iterrows():
            ax.annotate(
                str(int(row["latent_dim"])),
                (row["parameter_count"], row["psnr"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    ax.set_title(f"{dataset_name.upper()} Parameter vs PSNR Tradeoff")
    ax.set_xlabel("Stored Parameters / Weights")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_summary_markdown(
    metrics_df: pd.DataFrame, config: BenchmarkConfig, output_path: Path
) -> None:
    lines = [
        "# Unified Reconstruction Comparison",
        "",
        "This benchmark compares PCA, AE, DAE, and VAE on the same dataset split,",
        "using the same reconstruction metrics on normalized `0..1` pixels.",
        "",
        "Diffusion is excluded from the core table because it is not a latent",
        "dimensionality reduction model and therefore is not an apples-to-apples",
        "compression/reconstruction baseline against PCA or autoencoders.",
        "",
        "## Benchmark Setup",
        "",
        f"- datasets: {', '.join(config.datasets)}",
        f"- methods: {', '.join(config.methods)}",
        f"- latent dims: {', '.join(str(dim) for dim in config.latent_dims)}",
        f"- train subset size: {config.train_limit}",
        f"- test subset size: {config.test_limit}",
        f"- epochs for learned models: {config.epochs}",
        f"- batch size: {config.batch_size}",
        "",
    ]

    for dataset_name in config.datasets:
        dataset_df = metrics_df[metrics_df["dataset"] == dataset_name].copy()
        best_psnr = dataset_df.sort_values("psnr", ascending=False).iloc[0]
        best_ssim = dataset_df.sort_values("ssim", ascending=False).iloc[0]
        most_compact = dataset_df.sort_values(
            ["latent_dim", "psnr"], ascending=[True, False]
        ).iloc[0]
        lines.extend(
            [
                f"## {dataset_name.upper()}",
                "",
                f"- best PSNR: `{best_psnr['method']}` at latent `{int(best_psnr['latent_dim'])}` "
                f"with `{best_psnr['psnr']:.3f} dB`",
                f"- best SSIM: `{best_ssim['method']}` at latent `{int(best_ssim['latent_dim'])}` "
                f"with `{best_ssim['ssim']:.4f}`",
                f"- strongest low-dimension point in this run: `{most_compact['method']}` at latent "
                f"`{int(most_compact['latent_dim'])}` with `{most_compact['psnr']:.3f} dB`",
                "",
            ]
        )

        compact_df = dataset_df[
            [
                "method",
                "latent_dim",
                "mse",
                "mae",
                "psnr",
                "ssim",
                "parameter_count",
                "train_seconds",
            ]
        ].sort_values(["latent_dim", "method"])
        lines.append("```text")
        lines.append(compact_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        lines.append("```")
        lines.append("")

    output_path.write_text("\n".join(lines))


def run_pca(
    train_subset: Dataset,
    test_subset: Dataset,
    latent_dim: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    X_train = dataset_to_numpy(train_subset)
    X_test = dataset_to_numpy(test_subset)
    pca = PCA(n_components=latent_dim, svd_solver="randomized", random_state=seed)
    pca.fit(X_train)
    X_test_reconstructed = pca.inverse_transform(pca.transform(X_test)).astype(np.float32)
    X_test_reconstructed = np.clip(X_test_reconstructed, 0.0, 1.0)
    reconstructed_tensor = torch.from_numpy(X_test_reconstructed).view(-1, 1, 28, 28)
    metrics = {
        "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
    }
    return reconstructed_tensor, metrics


def run_learned_model(
    *,
    model_type: str,
    latent_dim: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    noise_level: float,
) -> tuple[nn.Module, float]:
    if model_type in ("ae", "dae"):
        model: nn.Module = FullyConnectedAutoencoder(latent_dim=latent_dim).to(device)
    elif model_type == "vae":
        model = VariationalAutoencoder(latent_dim=latent_dim).to(device)
    else:
        raise ValueError(f"Unsupported learned model type: {model_type}")

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            model_type=model_type,
            noise_level=noise_level,
        )
        eval_epoch(
            model,
            test_loader,
            criterion,
            device,
            model_type=model_type,
            noise_level=noise_level,
        )

    final_metrics = evaluate_metrics(
        model,
        test_loader,
        criterion,
        device,
        model_type=model_type,
    )
    return model, float(final_metrics["mse"])


def reconstruct_from_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_type: str,
) -> torch.Tensor:
    model.eval()
    batches = []
    with torch.no_grad():
        for clean_images, _ in test_loader:
            clean_images = clean_images.to(device)
            if model_type == "vae":
                reconstructed = model.reconstruct(clean_images)
            else:
                reconstructed = model(clean_images)
            batches.append(reconstructed.detach().cpu())
    return torch.cat(batches, dim=0)


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    directories = ensure_dirs(config.output_dir)

    rows: list[dict[str, object]] = []

    for dataset_index, dataset_name in enumerate(config.datasets):
        train_dataset = build_autoencoder_dataset(
            dataset_name, root=config.data_dir, train=True, download=config.download
        )
        test_dataset = build_autoencoder_dataset(
            dataset_name, root=config.data_dir, train=False, download=config.download
        )

        train_subset = select_subset(
            train_dataset, config.train_limit, config.seed + dataset_index * 101
        )
        test_subset = select_subset(
            test_dataset, config.test_limit, config.seed + dataset_index * 101 + 1
        )

        train_loader = make_loader(
            train_subset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            seed=config.seed,
            device=device,
        )
        test_loader = make_loader(
            test_subset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            seed=config.seed,
            device=device,
        )
        test_originals = dataset_to_tensor(test_subset)
        grid_reconstructions: dict[int, dict[str, torch.Tensor]] = {}

        for latent_dim in config.latent_dims:
            for method in config.methods:
                start = time.perf_counter()
                explained_variance_ratio = None

                if method == "pca":
                    reconstructed = None
                    reconstructed, pca_stats = run_pca(
                        train_subset, test_subset, latent_dim, config.seed
                    )
                    parameter_count = pca_storage_parameters(latent_dim)
                    explained_variance_ratio = pca_stats["explained_variance_ratio"]
                else:
                    model, _ = run_learned_model(
                        model_type=method,
                        latent_dim=latent_dim,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        epochs=config.epochs,
                        lr=config.lr,
                        device=device,
                        noise_level=config.noise_level,
                    )
                    reconstructed = reconstruct_from_model(
                        model, test_loader, device, method
                    )
                    parameter_count = count_parameters(model)

                train_seconds = time.perf_counter() - start
                metrics = compute_reconstruction_metrics(
                    test_originals, reconstructed, batch_size=config.batch_size
                )

                if latent_dim in DEFAULT_GRID_DIMS:
                    grid_reconstructions.setdefault(latent_dim, {})[method] = reconstructed

                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "latent_dim": latent_dim,
                        "train_limit": config.train_limit,
                        "test_limit": config.test_limit,
                        "epochs": 0 if method == "pca" else config.epochs,
                        "batch_size": config.batch_size,
                        "learning_rate": 0.0 if method == "pca" else config.lr,
                        "parameter_count": parameter_count,
                        "compression_ratio": INPUT_DIM / latent_dim,
                        "device": device.type,
                        "train_seconds": train_seconds,
                        "explained_variance_ratio": explained_variance_ratio,
                        **metrics,
                    }
                )

        for latent_dim, reconstructions_by_method in grid_reconstructions.items():
            save_method_grid(
                test_originals,
                reconstructions_by_method,
                directories["grids"]
                / f"{dataset_name}_latent_{latent_dim:03d}_method_comparison.png",
                title=f"{dataset_name.upper()} Reconstruction Comparison ({latent_dim}D)",
            )

    metrics_df = pd.DataFrame(rows).sort_values(["dataset", "latent_dim", "method"])
    metrics_df.to_csv(directories["metrics"] / "unified_metrics.csv", index=False)

    pivot = metrics_df.pivot_table(
        index=["dataset", "latent_dim"], columns="method", values=["psnr", "ssim", "mse"]
    )
    pivot.to_csv(directories["metrics"] / "metric_pivot.csv")

    for dataset_name in config.datasets:
        plot_metric_by_latent_dim(
            metrics_df,
            dataset_name=dataset_name,
            metric="psnr",
            output_path=directories["plots"] / f"{dataset_name}_psnr_vs_latent.png",
        )
        plot_metric_by_latent_dim(
            metrics_df,
            dataset_name=dataset_name,
            metric="ssim",
            output_path=directories["plots"] / f"{dataset_name}_ssim_vs_latent.png",
        )
        plot_metric_by_latent_dim(
            metrics_df,
            dataset_name=dataset_name,
            metric="mse",
            output_path=directories["plots"] / f"{dataset_name}_mse_vs_latent.png",
        )
        plot_tradeoff(
            metrics_df,
            dataset_name=dataset_name,
            output_path=directories["plots"] / f"{dataset_name}_parameter_tradeoff.png",
        )

    config_payload = asdict(config)
    config_payload["output_dir"] = str(config.output_dir)
    config_payload["data_dir"] = str(config.data_dir)
    (directories["metrics"] / "benchmark_config.json").write_text(
        json.dumps(config_payload, indent=2)
    )
    build_summary_markdown(
        metrics_df,
        config,
        directories["metrics"] / "summary.md",
    )

    print("Saved unified benchmark metrics to", directories["metrics"] / "unified_metrics.csv")
    print("Saved summary to", directories["metrics"] / "summary.md")
    print("Saved plots to", directories["plots"])
    print("Saved grids to", directories["grids"])


if __name__ == "__main__":
    main()
