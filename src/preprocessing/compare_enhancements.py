
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import random

ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_DIR = ROOT / "data/yolo_dataset/images/train"
ENHANCED_DIR = ROOT / "data/enhanced_dataset/images/train"

random.seed(42)

images = sorted(ORIGINAL_DIR.glob("*.jpg"))
samples = random.sample(images, min(6, len(images)))

fig, axes = plt.subplots(len(samples), 2, figsize=(14, 4 * len(samples)))

if len(samples) == 1:
    axes = [axes]

for i, image_path in enumerate(samples):

    original = cv2.imread(str(image_path))
    enhanced_path = ENHANCED_DIR / image_path.name
    enhanced = cv2.imread(str(enhanced_path))

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    axes[i][0].imshow(original)
    axes[i][0].set_title(f"Original: {image_path.name}")
    axes[i][0].axis("off")

    axes[i][1].imshow(enhanced)
    axes[i][1].set_title("Enhanced")
    axes[i][1].axis("off")

plt.tight_layout()
plt.show()
