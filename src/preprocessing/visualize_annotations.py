"""Visualize YOLO bounding boxes on prepared images."""

from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def display_yolo_annotations(
    dataset_dir: Path,
    split: str = "train",
    number: int = 4,
    seed: int = 42,
    image_stems=None,
):
    dataset_dir = Path(dataset_dir)
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing image or label folder for split {split!r}")

    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if image_stems is not None:
        requested = set(image_stems)
        image_files = [p for p in image_files if p.stem in requested]
    else:
        rng = random.Random(seed)
        image_files = rng.sample(image_files, min(number, len(image_files)))

    if not image_files:
        raise ValueError("No matching images to display.")

    fig, axes = plt.subplots(1, len(image_files), figsize=(6 * len(image_files), 6))
    if len(image_files) == 1:
        axes = [axes]

    for ax, image_path in zip(axes, image_files):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label: {label_path}")

        ax.imshow(image)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, xc, yc, box_w, box_h = map(float, line.split())
            x_center, y_center = xc * width, yc * height
            pixel_w, pixel_h = box_w * width, box_h * height
            xmin = x_center - pixel_w / 2
            ymin = y_center - pixel_h / 2

            ax.add_patch(
                patches.Rectangle(
                    (xmin, ymin), pixel_w, pixel_h,
                    linewidth=2, edgecolor="red", facecolor="none"
                )
            )

        ax.set_title(image_path.stem)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    display_yolo_annotations(
        project_root / "data" / "yolo_dataset",
        split="train",
        number=4,
    )
