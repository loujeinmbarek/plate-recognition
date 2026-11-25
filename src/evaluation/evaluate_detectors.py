
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = ROOT / "data.yaml"

MODELS = {
    "baseline": ROOT / "runs/yolo11n_baseline-2/weights/best.pt",
    "enhanced": ROOT / "runs/yolo11n_enhanced/weights/best.pt",
}

for name, model_path in MODELS.items():

    print(f"\n{'=' * 50}")
    print(f"FINAL TEST EVALUATION: {name.upper()}")
    print(f"{'=' * 50}")

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=4,
        device=0,
        project=str(ROOT / "runs" / "final_evaluation"),
        name=name,
        exist_ok=True,
        plots=True,
    )

    print(f"\n{name.upper()} TEST RESULTS")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

print("\nBoth models evaluated on the original test set.")
