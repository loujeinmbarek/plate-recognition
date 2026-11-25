
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = PROJECT_ROOT / "data.yaml"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        patience=15,
        pretrained=True,
        seed=42,
        project=str(PROJECT_ROOT / "runs"),
        name="yolo11n_baseline",
        plots=True
    )

if __name__ == "__main__":
    main()
