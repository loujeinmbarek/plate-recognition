
from pathlib import Path
import shutil
import cv2

from src.preprocessing.enhance_images import enhance_image

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/yolo_dataset"
DESTINATION = ROOT / "data/enhanced_dataset"

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def prepare_split(split):
    source_images = SOURCE / "images" / split
    source_labels = SOURCE / "labels" / split

    destination_images = DESTINATION / "images" / split
    destination_labels = DESTINATION / "labels" / split

    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in source_images.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    processed = 0

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")

        enhanced = enhance_image(image)

        output_image = destination_images / image_path.name

        if not cv2.imwrite(str(output_image), enhanced):
            raise RuntimeError(f"Cannot save {output_image}")

        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")

        shutil.copy2(
            label_path,
            destination_labels / label_path.name
        )

        processed += 1

    print(f"{split.upper()}: {processed} images prepared")


if __name__ == "__main__":
    for split in SPLITS:
        prepare_split(split)

    print("\nEnhanced dataset preparation completed!")
