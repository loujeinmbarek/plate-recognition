import cv2
import torch
import easyocr
from ultralytics import YOLO

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

model = YOLO("yolov8n.pt")  # downloads the pretrained weights on first run
print("YOLO loaded OK, number of classes:", len(model.names))

reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())  # downloads OCR models on first run
print("EasyOCR loaded OK")