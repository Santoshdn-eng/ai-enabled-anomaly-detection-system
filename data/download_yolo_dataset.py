"""
Lightweight Dataset Downloader & Pre-trained Model Fetcher
Optimized for 20 GB free SSD storage on Apple Silicon M4.
"""

import os
import sys

def setup_data_directories():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_dir = os.path.join(base_dir, "yolo_dataset")
    samples_dir = os.path.join(base_dir, "custom_samples")
    
    os.makedirs(os.path.join(yolo_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "labels", "val"), exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    print(f"✅ Data directories ready at: {base_dir}")
    print(f"  ├── yolo_dataset/ ({yolo_dir})")
    print(f"  └── custom_samples/ ({samples_dir})")

def fetch_pretrained_yolo_weights():
    """
    Downloads lightweight Ultralytics YOLOv8 weights (only ~6 MB).
    """
    try:
        from ultralytics import YOLO
        print("\n📥 Fetching lightweight YOLOv8 Nano weights (~6.2 MB)...")
        model = YOLO("yolov8n.pt")
        print("✅ YOLOv8 Nano weights downloaded and ready!")
        
        print("\n📥 Fetching lightweight YOLOv8 Pose weights (~6.5 MB)...")
        pose_model = YOLO("yolov8n-pose.pt")
        print("✅ YOLOv8 Pose weights downloaded and ready!")
    except ImportError:
        print("\n⚠️ 'ultralytics' package not installed yet. Install via: pip install ultralytics")

if __name__ == "__main__":
    print("=" * 60)
    print("AI Surveillance Anomaly System - Lightweight Dataset Setup")
    print("=" * 60)
    setup_data_directories()
    fetch_pretrained_yolo_weights()
