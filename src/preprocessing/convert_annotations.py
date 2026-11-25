"""Convert Pascal VOC XML annotations into normalized YOLO TXT labels."""

from pathlib import Path
import xml.etree.ElementTree as ET

IMAGE_SPLITS = {
    "train": "train",
    "val": "train",   # validation images are sampled from original/train
    "test": "test",
}


def convert_voc_to_yolo(xml_path: Path, output_path: Path, class_name: str = "LP"):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    image_width = int(size.findtext("width"))
    image_height = int(size.findtext("height"))
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}")

    rows = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name != class_name:
            continue

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing <bndbox> in {xml_path}")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid bounding box in {xml_path}: {xmin, ymin, xmax, ymax}")

        # Keep boundary-touching boxes valid; do not clip or silently alter them.
        if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
            raise ValueError(f"Bounding box outside image in {xml_path}")

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        rows.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    if not rows:
        raise ValueError(f"No objects of class {class_name!r} found in {xml_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def convert_split(original_dir: Path, output_dir: Path, split: str, overwrite: bool = False):
    if split not in IMAGE_SPLITS:
        raise ValueError(f"Unknown split: {split}")

    xml_source = Path(original_dir) / IMAGE_SPLITS[split]
    image_dir = Path(output_dir) / "images" / split
    label_dir = Path(output_dir) / "labels" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image split: {image_dir}")

    label_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        xml_path = xml_source / f"{image_path.stem}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(f"Missing source XML for {image_path.name}: {xml_path}")

        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not overwrite:
            skipped += 1
            continue

        convert_voc_to_yolo(xml_path, label_path)
        converted += 1

    print(f"{split.upper()}: {converted} converted, {skipped} existing labels kept")
    return converted, skipped


def convert_all(original_dir: Path, output_dir: Path, overwrite: bool = False):
    totals = {}
    for split in ("train", "val", "test"):
        totals[split] = convert_split(original_dir, output_dir, split, overwrite)
    return totals


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    convert_all(
        original_dir=project_root / "data" / "dataset",
        output_dir=project_root / "data" / "yolo_dataset",
    )
