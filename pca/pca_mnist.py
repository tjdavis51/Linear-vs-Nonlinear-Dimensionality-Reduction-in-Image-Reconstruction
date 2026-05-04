from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_CACHE_DIR = Path(".cache")
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_CACHE_DIR / "matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA


DEFAULT_COMPONENTS = [16, 32, 64, 128]
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
PIXEL_MAX = 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA image reconstruction experiments on MNIST."
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10000,
        help="Number of MNIST samples used to fit PCA.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2000,
        help="Number of MNIST samples used for reconstruction evaluation.",
    )
    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=DEFAULT_COMPONENTS,
        help="List of PCA component counts to evaluate.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pca/results"),
        help="Directory for metrics, plots, and reconstructed image grids.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    directories = {
        "root": output_dir,
        "metrics": output_dir / "metrics",
        "plots": output_dir / "plots",
        "reconstructions": output_dir / "reconstructions",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    data_home = PROJECT_CACHE_DIR / "scikit_learn_data"
    data_home.mkdir(parents=True, exist_ok=True)
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        data_home=data_home,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int16)
    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_size: int,
    test_size: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_requested = train_size + test_size
    if total_requested > len(X):
        raise ValueError(
            f"Requested {total_requested} samples, but MNIST only has {len(X)}."
        )

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(X))[:total_requested]
    selected_X = X[indices]
    selected_y = y[indices]

    X_train = selected_X[:train_size]
    X_test = selected_X[train_size:]
    y_test = selected_y[train_size:]
    return X_train, X_test, y_test


def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original - reconstructed) ** 2))


def mean_absolute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean(np.abs(original - reconstructed)))


def peak_signal_to_noise_ratio(
    original: np.ndarray, reconstructed: np.ndarray, pixel_max: float = PIXEL_MAX
) -> float:
    mse = mean_squared_error(original, reconstructed)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(pixel_max) - 10 * np.log10(mse))


def plot_metric_curves(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("mse", "Mean Squared Error"),
        ("rmse", "Root Mean Squared Error"),
        ("mae", "Mean Absolute Error"),
        ("psnr", "PSNR (dB)"),
    ]

    for ax, (column, title) in zip(axes.flat, metric_specs):
        ax.plot(metrics_df["n_components"], metrics_df[column], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Number of PCA Components")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("PCA Reconstruction Metrics on MNIST", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_explained_variance(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        metrics_df["n_components"],
        metrics_df["explained_variance_ratio"],
        marker="o",
        label="Variance Captured",
    )
    ax.set_title("Explained Variance by PCA Components")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_grid(
    originals: np.ndarray,
    reconstructed: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    sample_count: int = 10,
) -> None:
    fig, axes = plt.subplots(2, sample_count, figsize=(1.8 * sample_count, 4))
    fig.suptitle(title, fontsize=14)

    for idx in range(sample_count):
        original_ax = axes[0, idx]
        recon_ax = axes[1, idx]

        original_ax.imshow(
            originals[idx].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap="gray"
        )
        original_ax.set_title(f"Orig: {labels[idx]}")
        original_ax.axis("off")

        recon_ax.imshow(
            reconstructed[idx].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap="gray"
        )
        recon_ax.set_title("Recon")
        recon_ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    directories = ensure_output_dirs(args.output_dir)

    X, y = load_mnist()
    X_train, X_test, y_test = split_dataset(
        X=X,
        y=y,
        train_size=args.train_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    metrics: list[dict[str, float | int]] = []

    for n_components in args.components:
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=args.random_seed,
        )
        transformed = pca.fit_transform(X_train)

        reconstructed = pca.inverse_transform(pca.transform(X_test))
        reconstructed = np.clip(reconstructed, 0.0, PIXEL_MAX)
        
        _ = transformed

        mse = mean_squared_error(X_test, reconstructed)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(X_test, reconstructed)
        psnr = peak_signal_to_noise_ratio(X_test, reconstructed)
        explained_variance_ratio = float(np.sum(pca.explained_variance_ratio_))

        metrics.append(
            {
                "n_components": n_components,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "psnr": psnr,
                "explained_variance_ratio": explained_variance_ratio,
            }
        )

        save_reconstruction_grid(
            originals=X_test,
            reconstructed=reconstructed,
            labels=y_test,
            output_path=directories["reconstructions"]
            / f"reconstructions_{n_components:03d}_components.png",
            title=f"MNIST PCA Reconstruction ({n_components} Components)",
        )

    metrics_df = pd.DataFrame(metrics).sort_values("n_components").reset_index(
        drop=True
    )
    metrics_csv_path = directories["metrics"] / "pca_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    summary = {
        "dataset": "MNIST (OpenML mnist_784)",
        "image_shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "train_size": args.train_size,
        "test_size": args.test_size,
        "components": args.components,
        "metrics_csv": str(metrics_csv_path),
    }
    summary_path = directories["metrics"] / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_metric_curves(
        metrics_df=metrics_df,
        output_path=directories["plots"] / "reconstruction_metrics.png",
    )
    plot_explained_variance(
        metrics_df=metrics_df,
        output_path=directories["plots"] / "explained_variance.png",
    )

    print("Saved metrics to:", metrics_csv_path)
    print("Saved summary to:", summary_path)
    print("Saved plots to:", directories["plots"])
    print("Saved reconstruction grids to:", directories["reconstructions"])
    print()
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    run_experiment(parse_args())
