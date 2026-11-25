
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = ROOT / "data_enhanced.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

# Start from the same pretrained weights as Model A
model = YOLO("yolo11n.pt")

results = model.train(
    data=str(DATA_YAML),

    # Same training configuration as Model A
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    patience=15,
    pretrained=True,
    seed=42,
    plots=True,

    # No additional YOLO augmentation:
    # augmentation is already performed during preprocessing
    augment=False,

    # New experiment directory: preserve previous Model B
    project=str(ROOT / "runs"),
    name="yolo11n_enhanced_augmented",
)

print("\nEnhanced + augmented YOLO11n training completed!")
print(f"Results saved in: {results.save_dir}")
