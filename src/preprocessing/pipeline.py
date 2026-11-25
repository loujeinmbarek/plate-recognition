"""Safe entry point for preparing and validating the license-plate dataset."""

from pathlib import Path
import argparse

from split_dataset import split_dataset
from convert_annotations import convert_all
from validate_dataset import validate_dataset


def main():
    project_root = Path(__file__).resolve().parents[2]
    original_dir = project_root / "data" / "dataset"
    output_dir = project_root / "data" / "yolo_dataset"

    parser = argparse.ArgumentParser(description="Prepare the license-plate YOLO dataset.")
    parser.add_argument(
        "--create-split",
        action="store_true",
        help="Create image splits. Refuses to overwrite non-empty split folders.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Create missing YOLO labels; existing labels are kept.",
    )
    args = parser.parse_args()

    if args.create_split:
        split_dataset(original_dir, output_dir)

    if args.convert:
        convert_all(original_dir, output_dir, overwrite=False)

    # Default action, and final check after any requested stages.
    issues = validate_dataset(output_dir)
    if issues:
        raise SystemExit("Dataset validation failed. Review the issues above.")
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
