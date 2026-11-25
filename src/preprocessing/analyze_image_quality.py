
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"
CSV_PATH = REPORTS_DIR / "image_quality.csv"
PLOTS_DIR = REPORTS_DIR / "figures"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def analyze_image(image_path: Path, split: str) -> dict:
    """Calculate basic image-quality metrics without modifying the image."""

    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "split": split,
            "filename": image_path.name,
            "status": "unreadable",
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        "split": split,
        "filename": image_path.name,
        "status": "ok",
        "width": width,
        "height": height,
        "megapixels": round((width * height) / 1_000_000, 3),
        "aspect_ratio": round(width / height, 3),
        "brightness": round(float(np.mean(gray)), 2),
        "contrast": round(float(np.std(gray)), 2),
        "laplacian_variance": round(
            float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2
        ),
    }


def collect_image_paths():
    """Collect images from the original train and test folders."""

    image_paths = []

    for split in ("train", "test"):
        split_dir = DATASET_DIR / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {split_dir}")

        paths = sorted(
            path for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        image_paths.extend((path, split) for path in paths)

    return image_paths


def create_plots(df: pd.DataFrame):
    """Save distributions of the image-quality metrics."""

    valid = df[df["status"] == "ok"]

    if valid.empty:
        print("No readable images available for plotting.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("megapixels", "Image resolution (megapixels)"),
        ("brightness", "Brightness (mean grayscale intensity)"),
        ("contrast", "Contrast (grayscale standard deviation)"),
        ("laplacian_variance", "Sharpness indicator (Laplacian variance)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (column, title) in zip(axes.flat, metrics):
        ax.hist(valid[column].dropna(), bins=30, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Number of images")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    plot_path = PLOTS_DIR / "image_quality_distributions.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Plots saved to: {plot_path}")


def print_summary(df: pd.DataFrame):
    valid = df[df["status"] == "ok"]

    print("\n========== IMAGE QUALITY SUMMARY ==========")
    print(f"Total images:       {len(df)}")
    print(f"Readable images:    {len(valid)}")
    print(f"Unreadable images:  {(df['status'] == 'unreadable').sum()}")

    if valid.empty:
        return

    columns = [
        "width",
        "height",
        "megapixels",
        "brightness",
        "contrast",
        "laplacian_variance",
    ]

    print("\nMetric statistics:")
    print(valid[columns].describe().round(2).to_string())

    print("\nLowest sharpness-indicator values (inspect visually):")
    print(
        valid.nsmallest(10, "laplacian_variance")[
            ["split", "filename", "laplacian_variance"]
        ].to_string(index=False)
    )

    print("\nLowest brightness values (inspect visually):")
    print(
        valid.nsmallest(5, "brightness")[
            ["split", "filename", "brightness"]
        ].to_string(index=False)
    )

    print("\nHighest brightness values (inspect visually):")
    print(
        valid.nlargest(5, "brightness")[
            ["split", "filename", "brightness"]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze image quality in the original license-plate dataset."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate distribution plots.",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths()
    print(f"Found {len(image_paths)} images.")

    results = [
        analyze_image(image_path, split)
        for image_path, split in image_paths
    ]

    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)

    print(f"\nCSV report saved to: {CSV_PATH}")
    print_summary(df)

    if not args.no_plots:
        create_plots(df)


if __name__ == "__main__":
    main()
