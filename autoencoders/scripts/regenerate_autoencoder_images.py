#!/usr/bin/env python3
"""Regenerate final fully connected autoencoder reconstruction images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "image_reconstruction_matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from autoencoders import FullyConnectedAutoencoder, build_autoencoder_dataset, evaluate_metrics, train_epoch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = {
    "mnist": {
        "loader_name": "mnist",
        "output_name": "mnist",
        "label": "MNIST",
    },
    "fashion_mnist": {
        "loader_name": "fashion_mnist",
        "output_name": "fashion_mnist",
        "label": "Fashion-MNIST",
    },
}
DEFAULT_LATENT_DIMS = (2, 8, 16, 32, 64)
DEFAULT_SAMPLE_COUNT = 10
CRITERION = nn.MSELoss()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate autoencoder reconstruction PNGs.")
    parser.add_argument("--dataset", choices=("mnist", "fashion_mnist", "both"), default="both")
    parser.add_argument("--latent-dims", nargs="+", type=int, default=list(DEFAULT_LATENT_DIMS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("autoencoders/results"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("autoencoders/checkpoints"))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if any(latent_dim < 1 for latent_dim in args.latent_dims):
        parser.error("--latent-dims values must all be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not 8 <= args.samples <= 10:
        parser.error("--samples must be between 8 and 10")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


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


def selected_dataset_keys(dataset_arg: str) -> list[str]:
    if dataset_arg == "both":
        return ["mnist", "fashion_mnist"]
    return [dataset_arg]


def create_train_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def create_eval_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def load_test_samples(dataset: Dataset, sample_count: int, device: torch.device) -> torch.Tensor:
    images = [dataset[index][0] for index in range(sample_count)]
    return torch.stack(images, dim=0).to(device)


def checkpoint_candidates(checkpoint_dir: Path, dataset_name: str, latent_dim: int) -> tuple[Path, ...]:
    return (
        checkpoint_dir / dataset_name / f"latent_{latent_dim}.pt",
        checkpoint_dir / dataset_name / f"latent_{latent_dim}.pth",
        checkpoint_dir / f"{dataset_name}_latent_{latent_dim}.pt",
        checkpoint_dir / f"{dataset_name}_latent_{latent_dim}.pth",
    )


def extract_state_dict(payload: object) -> dict[str, torch.Tensor] | None:
    if isinstance(payload, dict):
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items()):
            return payload  # type: ignore[return-value]
    return None


def load_checkpoint_if_available(
    model: FullyConnectedAutoencoder,
    *,
    checkpoint_dir: Path,
    dataset_name: str,
    latent_dim: int,
    device: torch.device,
) -> Path | None:
    for candidate in checkpoint_candidates(checkpoint_dir, dataset_name, latent_dim):
        if not candidate.is_file():
            continue
        payload = torch.load(candidate, map_location=device)
        state_dict = extract_state_dict(payload)
        if state_dict is None:
            print(f"Skipping unsupported checkpoint format: {candidate}")
            continue
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            print(f"Skipping incompatible checkpoint {candidate}: {exc}")
            continue
        return candidate
    return None


def train_or_load_model(
    *,
    dataset_name: str,
    latent_dim: int,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[FullyConnectedAutoencoder, float | None, str]:
    model = FullyConnectedAutoencoder(latent_dim=latent_dim).to(device)
    checkpoint_path = load_checkpoint_if_available(
        model,
        checkpoint_dir=checkpoint_dir,
        dataset_name=dataset_name,
        latent_dim=latent_dim,
        device=device,
    )
    if checkpoint_path is not None:
        print(f"Loaded {dataset_name} latent {latent_dim} checkpoint: {checkpoint_path}")
        model.eval()
        return model, None, str(checkpoint_path)

    optimizer = Adam(model.parameters(), lr=lr)
    loss = None
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            CRITERION,
            device,
            model_type="ae",
            noise_level=0.0,
        )
        print(f"{dataset_name} latent {latent_dim} epoch {epoch}/{epochs} loss={loss:.6f}")
    model.eval()
    return model, loss, "trained"


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def reconstruct(model: FullyConnectedAutoencoder, images: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(images).detach().cpu().clamp(0.0, 1.0)


def save_reconstruction_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    dataset_label: str,
    latent_dim: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    originals = originals.detach().cpu().clamp(0.0, 1.0)
    reconstructions = reconstructions.detach().cpu().clamp(0.0, 1.0)
    sample_count = originals.shape[0]

    figure, axes = plt.subplots(2, sample_count, figsize=(1.35 * sample_count, 3.1), squeeze=False)
    rows = ((originals, "Original"), (reconstructions, "Reconstructed"))
    for row_index, (row_images, row_label) in enumerate(rows):
        for column_index in range(sample_count):
            axis = axes[row_index, column_index]
            axis.imshow(row_images[column_index].squeeze(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.axis("off")
        axes[row_index, 0].set_ylabel(row_label, rotation=0, labelpad=42, va="center", fontsize=9)

    figure.suptitle(f"{dataset_label} Autoencoder Reconstructions (latent dim = {latent_dim})", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_latent_comparison_grid(
    originals: torch.Tensor,
    reconstructions_by_latent: dict[int, torch.Tensor],
    *,
    dataset_label: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_dims = list(reconstructions_by_latent)
    originals = originals.detach().cpu().clamp(0.0, 1.0)
    sample_count = originals.shape[0]
    columns = len(latent_dims) + 1

    figure, axes = plt.subplots(sample_count, columns, figsize=(1.35 * columns, 1.25 * sample_count), squeeze=False)
    titles = ["Original", *[f"z={latent_dim}" for latent_dim in latent_dims]]
    for column_index, title in enumerate(titles):
        axes[0, column_index].set_title(title, fontsize=10)

    for row_index in range(sample_count):
        axes[row_index, 0].imshow(originals[row_index].squeeze(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[row_index, 0].axis("off")
        for column_index, latent_dim in enumerate(latent_dims, start=1):
            image = reconstructions_by_latent[latent_dim][row_index].squeeze()
            axes[row_index, column_index].imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axes[row_index, column_index].axis("off")

    figure.suptitle(f"{dataset_label} Autoencoder Latent Dimension Comparison", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


METRIC_FIELDNAMES = [
    "dataset",
    "latent_dim",
    "epochs",
    "batch_size",
    "learning_rate",
    "seed",
    "device",
    "model_parameters",
    "sample_count",
    "source",
    "final_train_loss",
    "test_mse",
    "test_psnr",
    "test_ssim",
]


def write_metrics(output_dir: Path, rows: list[dict[str, object]]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return [csv_path, json_path]


def process_dataset(args: argparse.Namespace, dataset_key: str, device: torch.device) -> tuple[list[Path], list[dict[str, object]]]:
    dataset_spec = DATASETS[dataset_key]
    loader_name = dataset_spec["loader_name"]
    output_name = dataset_spec["output_name"]
    dataset_label = dataset_spec["label"]
    dataset_output_dir = args.output_dir / output_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_autoencoder_dataset(loader_name, root=args.data_dir, train=True, download=args.download)
    test_dataset = build_autoencoder_dataset(loader_name, root=args.data_dir, train=False, download=args.download)
    originals = load_test_samples(test_dataset, args.samples, device)
    eval_loader = create_eval_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    saved_paths: list[Path] = []
    metric_rows: list[dict[str, object]] = []
    reconstructions_by_latent: dict[int, torch.Tensor] = {}

    for latent_dim in args.latent_dims:
        train_loader = create_train_loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed + latent_dim,
            device=device,
        )
        model, final_train_loss, source = train_or_load_model(
            dataset_name=output_name,
            latent_dim=latent_dim,
            train_loader=train_loader,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
        )
        metrics = evaluate_metrics(model, eval_loader, CRITERION, device, model_type="ae")
        reconstructions = reconstruct(model, originals)
        reconstructions_by_latent[latent_dim] = reconstructions
        latent_path = dataset_output_dir / f"latent_{latent_dim}.png"
        save_reconstruction_grid(
            originals,
            reconstructions,
            dataset_label=dataset_label,
            latent_dim=latent_dim,
            output_path=latent_path,
        )
        saved_paths.append(latent_path)
        print(f"Saved {latent_path}")
        metric_rows.append(
            {
                "dataset": output_name,
                "latent_dim": latent_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "seed": args.seed,
                "device": str(device),
                "model_parameters": count_trainable_parameters(model),
                "sample_count": args.samples,
                "source": source,
                "final_train_loss": final_train_loss,
                "test_mse": metrics["mse"],
                "test_psnr": metrics["psnr"],
                "test_ssim": metrics["ssim"],
            }
        )
        print(
            f"{output_name} latent {latent_dim} metrics: "
            f"mse={metrics['mse']:.6f} psnr={metrics['psnr']:.3f} ssim={metrics['ssim']:.4f}"
        )

    comparison_path = dataset_output_dir / "latent_comparison_grid.png"
    save_latent_comparison_grid(
        originals,
        reconstructions_by_latent,
        dataset_label=dataset_label,
        output_path=comparison_path,
    )
    saved_paths.append(comparison_path)
    print(f"Saved {comparison_path}")
    metrics_paths = write_metrics(dataset_output_dir, metric_rows)
    saved_paths.extend(metrics_paths)
    for path in metrics_paths:
        print(f"Saved {path}")
    return saved_paths, metric_rows


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    saved_paths: list[Path] = []
    metric_rows: list[dict[str, object]] = []
    for dataset_key in selected_dataset_keys(args.dataset):
        dataset_paths, dataset_metrics = process_dataset(args, dataset_key, device)
        saved_paths.extend(dataset_paths)
        metric_rows.extend(dataset_metrics)

    combined_metrics_paths = write_metrics(args.output_dir, metric_rows)
    saved_paths.extend(combined_metrics_paths)
    for path in combined_metrics_paths:
        print(f"Saved {path}")

    print("Regenerated outputs:")
    for path in saved_paths:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
