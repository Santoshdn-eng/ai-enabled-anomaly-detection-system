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
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import asyncio

from backend.model_engine import AnomalyDetectorEngine
from backend.cctv_streamer import CCTVStreamManager

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

# Initialize Engine & CCTV Stream Manager
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.keras")
HF_REPO = os.getenv("HF_REPO_ID", "SantoshDN/ai-enable-anomaly-detection")

engine = AnomalyDetectorEngine(model_path=MODEL_PATH, hf_repo_id=HF_REPO)
cctv_manager = CCTVStreamManager(engine)

# Serve Frontend static files if available
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

class CCTVConnectRequest(BaseModel):
    camera_id: str = Field(..., example="cam_02")
    name: str = Field(..., example="North Entrance CCTV")
    source_url: str = Field(..., example="rtsp://admin:pass@192.168.1.120:554/stream1")

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
        "classes": engine.classes,
        "cctv_streams": len(cctv_manager.cameras)
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
async def predict_live_frame(file: UploadFile = File(...), sensitivity: float = 0.60):
    """
    Real-time Live WebCam / frame snapshot inference endpoint.
    Performs YOLOv8 Person/Object detection and Deep Learning anomaly analysis.
    """
    try:
        contents = await file.read()
        result = engine.process_frame_image(contents, sensitivity_threshold=sensitivity)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live frame inference error: {str(e)}")

# --- CCTV STREAM MANAGER ENDPOINTS ---

@app.get("/api/cctv/streams")
def list_cctv_streams():
    """Returns list of active CCTV camera stream workers and their current live threat status."""
    return cctv_manager.list_cameras()

@app.post("/api/cctv/connect")
def connect_cctv_stream(req: CCTVConnectRequest):
    """Connect to a new RTSP, RTMP, HTTP/MJPEG, or Webcam stream URL globally."""
    try:
        worker = cctv_manager.add_camera(req.camera_id, req.name, req.source_url)
        return {
            "status": "success",
            "message": f"CCTV Stream worker '{req.name}' connected.",
            "camera": worker.get_status_info()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not connect CCTV stream: {e}")

@app.delete("/api/cctv/disconnect/{camera_id}")
def disconnect_cctv_stream(camera_id: str):
    """Disconnect and remove an active CCTV stream."""
    if camera_id not in cctv_manager.cameras:
        raise HTTPException(status_code=404, detail="Camera ID not found")
    cctv_manager.remove_camera(camera_id)
    return {"status": "success", "message": f"Camera '{camera_id}' disconnected."}

@app.get("/api/cctv/mjpeg/{camera_id}")
def get_cctv_mjpeg_stream(camera_id: str):
    """
    Streams live annotated MJPEG video from configured CCTV camera stream directly to browser / video tags.
    """
    worker = cctv_manager.get_camera(camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail="CCTV Camera not found")

    def mjpeg_generator():
        while worker.is_running:
            jpg_bytes = worker.get_latest_mjpeg_bytes()
            if jpg_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
            time.sleep(0.04)

    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/cctv/{camera_id}")
async def websocket_cctv_stream(websocket: WebSocket, camera_id: str):
    """
    Continuous WebSocket endpoint streaming frame-by-frame JSON detection results,
    YOLOv8 bounding boxes, and pose keypoint graphics for a specific CCTV camera feed.
    """
    await websocket.accept()
    worker = cctv_manager.get_camera(camera_id)
    if not worker:
        await websocket.send_json({"error": f"Camera {camera_id} not found"})
        await websocket.close()
        return

    print(f"⚡ [WebSocket] Client listening to CCTV stream '{camera_id}'")
    try:
        last_sent_time = 0.0
        while worker.is_running:
            if worker.last_update_time > last_sent_time and worker.latest_detection:
                last_sent_time = worker.last_update_time
                await websocket.send_json(worker.latest_detection)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print(f"⚡ [WebSocket] Client disconnected from CCTV '{camera_id}'")
    except Exception as e:
        print(f"[WebSocket] Error on CCTV '{camera_id}': {e}")
        try:
            await websocket.close()
        except Exception:
            pass

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
async def websocket_live_detection(websocket: WebSocket, sensitivity: float = 0.60):
    """
    Real-time Interactive WebSocket endpoint for continuous live camera frame analysis.
    Clients stream binary frame images (JPEG) and receive instant anomaly predictions & YOLO graphics.
    """
    await websocket.accept()
    print("⚡ [WebSocket] Client connected to live camera detection stream.")
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue
            # Perform live frame inference asynchronously in worker thread
            result = await asyncio.to_thread(engine.process_frame_image, data, sensitivity_threshold=sensitivity)
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

