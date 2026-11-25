
from pathlib import Path
import cv2
import argparse


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data/yolo_dataset/images/train"
OUTPUT_DIR = ROOT / "data/enhanced_dataset/images/train"


def enhance_image(image):
    """
    Apply controlled image enhancement:
    1. CLAHE contrast enhancement
    2. Bilateral noise reduction
    3. Unsharp masking
    """

    # 1. Improve local contrast in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    contrast_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # 2. Reduce noise while preserving edges
    denoised = cv2.bilateralFilter(
        contrast_image,
        d=5,
        sigmaColor=50,
        sigmaSpace=50
    )

    # 3. Sharpen details
    blurred = cv2.GaussianBlur(
        denoised,
        (0, 0),
        sigmaX=1.0
    )

    sharpened = cv2.addWeighted(
        denoised, 1.5,
        blurred, -0.5,
        0
    )

    return sharpened


def process_dataset(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob("*"))

    image_paths = [
        p for p in image_paths
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    processed = 0
    failed = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            failed += 1
            continue

        enhanced = enhance_image(image)

        output_path = output_dir / image_path.name

        success = cv2.imwrite(str(output_path), enhanced)

        if success:
            processed += 1
        else:
            failed += 1

    print("\nEnhancement completed!")
    print(f"Images processed: {processed}")
    print(f"Images failed: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_DIR
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR
    )

    args = parser.parse_args()

    process_dataset(args.input, args.output)
