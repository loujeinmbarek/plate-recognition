
from pathlib import Path
import random
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SOURCE_IMAGES = ROOT / "data/enhanced_dataset/images/train"
SOURCE_LABELS = ROOT / "data/enhanced_dataset/labels/train"

DEST_IMAGES = ROOT / "data/augmented_dataset/images/train"
DEST_LABELS = ROOT / "data/augmented_dataset/labels/train"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

AUGMENTATIONS_PER_IMAGE = 2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def transform_image_and_boxes(image, labels):
    """
    Apply rotation, scaling and translation to an image.
    Apply the exact same transformation to its YOLO bounding boxes.
    """

    height, width = image.shape[:2]

    angle = random.uniform(-5, 5)
    scale = random.uniform(0.90, 1.10)
    tx = random.uniform(-0.05, 0.05) * width
    ty = random.uniform(-0.05, 0.05) * height

    center = (width / 2, height / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )

    transformed_labels = []

    for label in labels:
        class_id, x_center, y_center, box_width, box_height = label

        # Convert normalized YOLO coordinates to pixel coordinates.
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        # Transform all four bounding-box corners.
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1],
        ])

        transformed_corners = corners @ matrix.T

        new_x1 = np.clip(transformed_corners[:, 0].min(), 0, width)
        new_y1 = np.clip(transformed_corners[:, 1].min(), 0, height)
        new_x2 = np.clip(transformed_corners[:, 0].max(), 0, width)
        new_y2 = np.clip(transformed_corners[:, 1].max(), 0, height)

        new_width = new_x2 - new_x1
        new_height = new_y2 - new_y1

        # Discard boxes that are too small after transformation.
        if new_width < 2 or new_height < 2:
            continue

        new_x_center = (new_x1 + new_x2) / 2
        new_y_center = (new_y1 + new_y2) / 2

        transformed_labels.append([
            class_id,
            new_x_center / width,
            new_y_center / height,
            new_width / width,
            new_height / height,
        ])

    # Mild brightness and contrast variation.
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)

    transformed_image = cv2.convertScaleAbs(
        transformed_image,
        alpha=alpha,
        beta=beta,
    )

    return transformed_image, transformed_labels


def read_yolo_labels(label_path):
    labels = []

    if not label_path.exists():
        raise FileNotFoundError(f"Missing label: {label_path}")

    with open(label_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Invalid annotation in {label_path}, "
                    f"line {line_number}"
                )

            class_id = values[0]
            coordinates = [float(value) for value in values[1:]]

            labels.append([class_id, *coordinates])

    return labels


def save_yolo_labels(label_path, labels):
    with open(label_path, "w") as file:
        for label in labels:
            class_id = label[0]
            coordinates = label[1:]

            file.write(
                f"{class_id} "
                + " ".join(f"{value:.6f}" for value in coordinates)
                + "\n"
            )


def augment_dataset():

    if not SOURCE_IMAGES.exists() or not SOURCE_LABELS.exists():
        raise FileNotFoundError("Enhanced training dataset not found.")

    DEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_LABELS.mkdir(parents=True, exist_ok=True)

    images = sorted([
        path for path in SOURCE_IMAGES.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        raise RuntimeError("No training images found.")

    original_count = 0
    augmented_count = 0

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        labels = read_yolo_labels(label_path)

        # Keep the original enhanced image and its annotations.
        original_output = DEST_IMAGES / image_path.name
        original_label_output = DEST_LABELS / label_path.name

        if not cv2.imwrite(str(original_output), image):
            raise RuntimeError(f"Cannot save {original_output}")

        save_yolo_labels(original_label_output, labels)

        original_count += 1

        # Generate augmented copies.
        for index in range(1, AUGMENTATIONS_PER_IMAGE + 1):

            augmented_image, augmented_labels = (
                transform_image_and_boxes(image, labels)
            )

            if not augmented_labels:
                raise RuntimeError(
                    f"All boxes disappeared for {image_path.name}"
                )

            output_name = (
                f"{image_path.stem}_aug_{index:02d}"
                f"{image_path.suffix}"
            )

            output_image = DEST_IMAGES / output_name
            output_label = DEST_LABELS / f"{Path(output_name).stem}.txt"

            if not cv2.imwrite(str(output_image), augmented_image):
                raise RuntimeError(f"Cannot save {output_image}")

            save_yolo_labels(output_label, augmented_labels)

            augmented_count += 1

        if (original_count + augmented_count) % 100 == 0:
            print(
                f"Processed: {original_count} original, "
                f"{augmented_count} augmented"
            )

    print("\nDATA AUGMENTATION COMPLETED!")
    print(f"Original images copied: {original_count}")
    print(f"Augmented images created: {augmented_count}")
    print(f"Total training images: {original_count + augmented_count}")
    print(f"Output: {DEST_IMAGES}")


if __name__ == "__main__":
    augment_dataset()
