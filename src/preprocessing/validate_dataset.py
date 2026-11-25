"""Validate YOLO image/label pairs and normalized bounding boxes."""
import argparse
from pathlib import Path
import math


def validate_dataset(dataset_dir: Path, tolerance: float = 1e-6):
    dataset_dir = Path(dataset_dir)
    all_issues = []

    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split

        if not image_dir.is_dir() or not label_dir.is_dir():
            all_issues.append((split, "-", "Missing images or labels directory"))
            continue

        images = {
            p.stem for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        }
        labels = {p.stem for p in label_dir.glob("*.txt")}

        for stem in sorted(images - labels):
            all_issues.append((split, f"{stem}", "Missing label"))
        for stem in sorted(labels - images):
            all_issues.append((split, f"{stem}", "Missing image"))

        for label_path in sorted(label_dir.glob("*.txt")):
            lines = label_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                all_issues.append((split, label_path.name, "Empty annotation"))
                continue

            for line_no, line in enumerate(lines, start=1):
                values = line.split()
                if len(values) != 5:
                    all_issues.append((split, label_path.name, f"Line {line_no}: expected 5 values"))
                    continue

                try:
                    class_value, xc, yc, width, height = map(float, values)
                except ValueError:
                    all_issues.append((split, label_path.name, f"Line {line_no}: non-numeric value"))
                    continue

                if not all(math.isfinite(v) for v in (class_value, xc, yc, width, height)):
                    issue = f"Line {line_no}: non-finite value"
                elif class_value != 0:
                    issue = f"Line {line_no}: incorrect class ID (expected 0)"
                elif not (0 < xc < 1 and 0 < yc < 1):
                    issue = f"Line {line_no}: invalid center coordinates"
                elif not (0 < width <= 1 and 0 < height <= 1):
                    issue = f"Line {line_no}: invalid box dimensions"
                elif xc - width / 2 < -tolerance or xc + width / 2 > 1 + tolerance:
                    issue = f"Line {line_no}: box exceeds horizontal boundaries"
                elif yc - height / 2 < -tolerance or yc + height / 2 > 1 + tolerance:
                    issue = f"Line {line_no}: box exceeds vertical boundaries"
                else:
                    continue

                all_issues.append((split, label_path.name, issue))

    print("Validation issues:", len(all_issues))
    for issue in all_issues[:30]:
        print(issue)

    return all_issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/yolo_dataset"),
        help="Dataset directory to validate"
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / args.dataset

    issues = validate_dataset(dataset_dir)

    raise SystemExit(1 if issues else 0)
