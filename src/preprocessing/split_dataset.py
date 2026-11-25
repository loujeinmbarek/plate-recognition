"""Create train/validation/test image splits without modifying the source dataset."""

from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _images(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_dataset(
    original_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.20,
    random_seed: int = 42,
):
    """
    Split original_dir/train into train and val; preserve original_dir/test as test.
    Copies images only. XML files stay safely in the original dataset and are
    read directly by convert_annotations.py.
    Refuses to overwrite an existing prepared image split.
    """
    original_dir = Path(original_dir)
    output_dir = Path(output_dir)

    source_train = original_dir / "train"
    source_test = original_dir / "test"

    if not source_train.is_dir() or not source_test.is_dir():
        raise FileNotFoundError(
            f"Expected source folders: {source_train} and {source_test}"
        )

    destinations = {
        split: output_dir / "images" / split
        for split in ("train", "val", "test")
    }

    existing = [
        folder for folder in destinations.values()
        if folder.exists() and any(folder.iterdir())
    ]
    if existing:
        raise FileExistsError(
            "Prepared image folders already contain files. "
            "They were not changed. To preserve the existing split, do not "
            "run splitting again. Existing folders: "
            + ", ".join(map(str, existing))
        )

    train_images = _images(source_train)
    test_images = _images(source_test)

    if not train_images or not test_images:
        raise ValueError("The original train or test folder contains no images.")

    train_files, val_files = train_test_split(
        train_images,
        test_size=val_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    split_files = {
        "train": train_files,
        "val": val_files,
        "test": test_images,
    }

    for split, destination in destinations.items():
        destination.mkdir(parents=True, exist_ok=True)
        for image_path in split_files[split]:
            shutil.copy2(image_path, destination / image_path.name)

    counts = {split: len(items) for split, items in split_files.items()}
    print("Image split created:", counts)
    return counts


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    split_dataset(
        original_dir=project_root / "data" / "dataset",
        output_dir=project_root / "data" / "yolo_dataset",
    )
