# Dataset Management & Pre-trained Model Setup

## Storage Constraints Summary
- **Available Storage**: ~20 GB SSD on Apple Silicon M4 MacBook Air.
- **Strategy**: Use **Pre-trained Weights** (6MB - 25MB) + **Targeted Lightweight Subsets** (< 1.5 GB) instead of downloading massive 100+ GB raw video datasets.

---

## Recommended Lightweight Complementary Datasets

### 1. Roboflow Violence & Weapon Detection Dataset (YOLO Annotations)
- **Size**: ~500 MB - 1.2 GB
- **Format**: YOLOv8 PyTorch format (`images/train`, `labels/train`)
- **Classes**: `violence`, `person`, `weapon`, `knife`, `fire`, `normal`
- **Source**: Roboflow Universe (`roboflow` Python package)

### 2. RWF-2000 (Real-World Violence Dataset - Sample Subset)
- **Size**: ~1.5 GB
- **Format**: Short 5-second 30fps clips
- **Classes**: `Fight`, `Non-Fight`

---

## Pre-Trained Weights (Zero Extra Disk Usage)

Instead of training from scratch, utilize official pre-trained weights:
1. **YOLOv8 Nano (`yolov8n.pt`)**: ~6.2 MB
2. **YOLOv8 Pose (`yolov8n-pose.pt`)**: ~6.5 MB (Detects human keypoints to identify fighting/falling actions)
3. **Ano-AAD Hugging Face Model**: Pre-trained weights hosted on Hugging Face Hub.

---

## Directory Structure
```
data/
├── yolo_dataset/         # Lightweight YOLO images & bounding boxes
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── custom_samples/       # Test video clips
```
