
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "reports" / "image_quality.csv"
DATASET_PATH = PROJECT_ROOT / "data" / "dataset"


def display_images(df, metric, title, ascending=True, count=10):
    selected = df.sort_values(metric, ascending=ascending).head(count)

    columns = 5
    rows = (len(selected) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(18, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (_, item) in zip(axes, selected.iterrows()):
        image_path = DATASET_PATH / item["split"] / item["filename"]
        image = cv2.imread(str(image_path))

        if image is None:
            ax.set_title(f"{item['filename']} - unreadable")
            ax.axis("off")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

        ax.set_title(
            f"{item['split']}/{item['filename']}\n"
            f"{metric}: {item[metric]:.2f}"
        )
        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV report not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} image records.")

    display_images(
        df,
        metric="laplacian_variance",
        title="10 Images with Lowest Sharpness Indicator",
        ascending=True,
        count=10,
    )

    display_images(
        df,
        metric="brightness",
        title="5 Darkest Images",
        ascending=True,
        count=5,
    )

    display_images(
        df,
        metric="brightness",
        title="5 Brightest Images",
        ascending=False,
        count=5,
    )


if __name__ == "__main__":
    main()
