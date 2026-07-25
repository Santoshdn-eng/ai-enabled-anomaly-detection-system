# AI Enabled Anomaly Detection System

A complete full-stack Deep Learning & Surveillance system for detecting anomalous activities (`Arrest`, `Ill-treatment`, `Explosion`, `violence`, `Traffic Irregularities`, `Attack`, `Burglary`, `Fighting`, `fire-raising`, `Normal Videos`) from Kaggle Ano-AAD dataset.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Backend API & Dashboard Server
```bash
python -m backend.main
```
Navigate to **http://localhost:8000** in your browser to access the live surveillance dashboard!

---

## 📦 Hugging Face Model Hub Integration

### Pushing Trained Model to Hugging Face
1. Authenticate with your Hugging Face API key:
   ```bash
   huggingface-cli login
   ```
2. Upload your trained model file (`best_model.keras`):
   ```bash
   python hf_integration/push_to_hf.py
   ```

### Pulling Model in Production / API
Use the included helper script:
```python
from hf_integration.pull_from_hf import pull_model_from_hf

model_path = pull_model_from_hf("your-username/ai-enable-anomaly-detection")
```

---

## 📈 Model Improvement Guide

Key fixes implemented over the initial Kaggle notebook (~16% accuracy):
1. **Pretrained Backbone**: Switched from `weights=None` to `weights='imagenet'` on EfficientNetB0.
2. **Resolution**: Upgraded frame spatial size from 64x64 to 128x128.
3. **Data Augmentations**: Spatial flipping & temporal stride sampling.
4. **Learning Rate**: Fine-tuning with Adam `1e-4` + `ReduceLROnPlateau`.

To train the improved model:
```bash
python model/train_improved.py
```
