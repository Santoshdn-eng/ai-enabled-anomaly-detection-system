"""
FastAPI Server for AI Enabled Anomaly Detection Backend.
Provides REST endpoints for:
1. Video Analysis (/api/predict)
2. Camera Stream Simulation (/api/stream/simulate)
3. Model Health Check & Hugging Face Status (/api/health)
"""

import os
import random
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse, FileResponse

from backend.model_engine import AnomalyDetectorEngine

app = FastAPI(
    title="AI Enabled Anomaly Detection API",
    description="Backend service for detecting surveillance anomalies in video feeds using Deep Learning & Transformers.",
    version="1.0.0"
)

# Enable CORS for local & remote frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.keras")
HF_REPO = os.getenv("HF_REPO_ID", "SantoshDN/ai-enable-anomaly-detection")

engine = AnomalyDetectorEngine(model_path=MODEL_PATH, hf_repo_id=HF_REPO)

# Serve Frontend static files if available
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def read_root():
    """Serves the dashboard UI or API metadata"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "service": "AI Enabled Anomaly Detection API",
        "docs": "/docs"
    }

@app.get("/style.css")
def get_style():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"), media_type="text/css")

@app.get("/app.js")
def get_script():
    return FileResponse(os.path.join(FRONTEND_DIR, "app.js"), media_type="application/javascript")

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": engine.model is not None,
        "hf_repo": engine.hf_repo_id,
        "classes": engine.classes
    }

@app.post("/api/predict")
async def predict_anomaly(file: UploadFile = File(...)):
    """
    Upload a video clip (.mp4, .avi, .mov) for Anomaly Detection Analysis.
    Returns predicted class, confidence, frame-by-frame risk timeline, and probabilities.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid video file.")

    try:
        contents = await file.read()
        result = engine.process_video_file(contents, file.filename)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/predict_frame")
async def predict_live_frame(file: UploadFile = File(...)):
    """
    Real-time Live WebCam frame snapshot inference endpoint.
    Performs YOLOv8 Person/Object detection and Deep Learning anomaly analysis.
    """
    try:
        contents = await file.read()
        result = engine.process_frame_image(contents)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live frame inference error: {str(e)}")

@app.get("/api/model/info")
def get_model_info():
    """Returns detailed technical specification and status of the linked AI Model."""
    return {
        "status": "online",
        "hf_repo_id": engine.hf_repo_id,
        "is_model_loaded": engine.model is not None,
        "is_yolo_loaded": engine.yolo_model is not None,
        "input_resolution": f"{engine.img_size}x{engine.img_size}",
        "sequence_length": engine.seq_len,
        "total_classes": len(engine.classes),
        "supported_classes": engine.classes,
        "architecture": "MobileNetV2 / EfficientNet + TimeDistributed + LSTM + YOLOv8"
    }

@app.get("/api/model/metrics")
def get_model_classification_report():
    """
    Returns complete dataset evaluation metrics (Accuracy, Macro Avg, Weighted Avg, Precision, Recall, F1-score, Support).
    """
    return engine.get_model_classification_metrics()

@app.websocket("/ws/live")
async def websocket_live_detection(websocket: WebSocket):
    """
    Real-time Interactive WebSocket endpoint for continuous live camera frame analysis.
    Clients stream binary frame images (JPEG) and receive instant anomaly predictions.
    """
    await websocket.accept()
    print("⚡ [WebSocket] Client connected to live camera detection stream.")
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue
            # Perform live frame inference
            result = engine.process_frame_image(data)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("⚡ [WebSocket] Client disconnected from live stream.")
    except Exception as e:
        print(f"[WebSocket] Stream error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
