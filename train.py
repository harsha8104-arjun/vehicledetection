import torch
from ultralytics import YOLO
import os
import shutil

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Training parameters
train_params = {
    "data": "/kaggle/car-number-plate-dataset-yolo-format/License-Plate-Data/data.yaml",
    "epochs": 300,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "SGD",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "augment": True,
    "patience": 30,
    "device": 0 if torch.cuda.is_available() else 'cpu',
    "name": "lp_detector_highacc",
    "exist_ok": True
}

# Train the model
results=model.train(**train_params)

# Validate
metrics = model.val(data=train_params["data"], imgsz=train_params["imgsz"], device=train_params["device"])
print("Evaluation Results:", metrics)
