import torch
from ultralytics import YOLO
import cv2
import sys

print("Loading YOLO...")
model = YOLO('yolov8n.pt', task='detect')

print("Checking available cameras...")
for i in range(2):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i} is available and reading frames. Running YOLO...")
            try:
                results = model(frame, verbose=False)
                print("YOLO inference successful!")
            except Exception as e:
                print(f"YOLO ERROR: {e}")
        else:
            print(f"Camera {i} is available but CANNOT read frames.")
        cap.release()
    else:
        print(f"Camera {i} is not accessible.")

