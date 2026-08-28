"""
CCTV Live Capture Stream Manager.
Manages multi-camera RTSP/HTTP/RTMP/Webcam feeds, performs real-time continuous
frame grabbing, runs AI anomaly detection + YOLOv8 graphics extraction,
and streams live MJPEG video and WebSockets to connected surveillance dashboards.
"""

import time
import threading
import cv2
import numpy as np
import base64
from typing import Dict, Optional, List, Set
from backend.model_engine import AnomalyDetectorEngine

class CCTVCameraWorker:
    def __init__(self, camera_id: str, name: str, source_url: str, engine: AnomalyDetectorEngine):
        self.camera_id = camera_id
        self.name = name
        self.source_url = source_url
        self.engine = engine
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_detection: Optional[dict] = None
        self.last_update_time: float = 0.0
        self.status = "disconnected"
        self.subscribers: Set[object] = set()
        self.sensitivity = 0.60

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.status = "connecting"
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.status = "disconnected"
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _capture_loop(self):
        print(f"📹 [CCTV Worker] Starting camera stream: '{self.name}' ({self.source_url})")
        
        # Determine source (int if numeric string for local webcam)
        source = int(self.source_url) if self.source_url.isdigit() else self.source_url

        retry_count = 0
        while self.is_running:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.status = "error"
                print(f"[CCTV Worker] Failed to open stream '{self.name}'. Retrying in 3s...")
                time.sleep(3.0)
                retry_count += 1
                if retry_count > 10:
                    break
                continue

            self.status = "online"
            retry_count = 0
            frame_counter = 0

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"[CCTV Worker] Stream lost for '{self.name}'. Reconnecting...")
                    break

                frame_counter += 1
                # Downsample frame processing to ~10 FPS for optimal AI throughput
                if frame_counter % 2 == 0:
                    try:
                        # Process AI Anomaly Detection & YOLOv8 Graphics
                        detection = self.engine.process_cv2_frame(frame, sensitivity_threshold=self.sensitivity)
                        detection["camera_id"] = self.camera_id
                        detection["camera_name"] = self.name
                        
                        # Generate OpenCV annotated frame for direct MJPEG viewing
                        annotated = frame.copy()
                        yolo_gfx = detection.get("yolo_graphics", {})
                        boxes = yolo_gfx.get("boxes", [])
                        is_anomaly = detection.get("is_anomaly", False)
                        pred_class = detection.get("predicted_class", "Normal")
                        threat_pct = detection.get("threat_severity_pct", 0.0)

                        # Draw YOLO Bounding Boxes
                        for b in boxes:
                            x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
                            cls_name = b["class_name"]
                            conf = b["confidence"]
                            color = (0, 0, 255) if is_anomaly else (0, 255, 0)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated, f"{cls_name} {int(conf*100)}%", (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Draw Threat Alert Header
                        banner_color = (0, 0, 220) if is_anomaly else (0, 180, 0)
                        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), banner_color, -1)
                        status_text = f"CAM: {self.name} | THREAT: {threat_pct}% | CLASS: {pred_class.upper()}"
                        cv2.putText(annotated, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        with self.lock:
                            self.latest_frame = frame
                            self.latest_annotated_frame = annotated
                            self.latest_detection = detection
                            self.last_update_time = time.time()
                    except Exception as e:
                        print(f"[CCTV Worker] Processing error on '{self.name}': {e}")

                time.sleep(0.03) # Cap loop speed to ~30 FPS input

            cap.release()
            self.status = "reconnecting"
            time.sleep(1.5)

        self.status = "disconnected"
        print(f"📹 [CCTV Worker] Stopped camera stream: '{self.name}'")

    def get_latest_mjpeg_bytes(self) -> Optional[bytes]:
        with self.lock:
            frame = self.latest_annotated_frame if self.latest_annotated_frame is not None else self.latest_frame
        if frame is None:
            return None
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            return None
        return jpeg.tobytes()

    def get_status_info(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source_url": self.source_url,
            "status": self.status,
            "last_update": time.strftime("%H:%M:%S", time.localtime(self.last_update_time)) if self.last_update_time > 0 else "N/A",
            "latest_detection": self.latest_detection
        }

class CCTVStreamManager:
    def __init__(self, engine: AnomalyDetectorEngine):
        self.engine = engine
        self.cameras: Dict[str, CCTVCameraWorker] = {}
        # Cameras start empty so webcam hardware remains OFF by default until requested.

    def add_camera(self, camera_id: str, name: str, source_url: str) -> CCTVCameraWorker:
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
        worker = CCTVCameraWorker(camera_id, name, source_url, self.engine)
        self.cameras[camera_id] = worker
        worker.start()
        return worker

    def stop_all(self):
        """Stops all active camera workers and releases video capture hardware."""
        for worker in list(self.cameras.values()):
            worker.stop()
        self.cameras.clear()

    def remove_camera(self, camera_id: str):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]

    def list_cameras(self) -> List[dict]:
        return [worker.get_status_info() for worker in self.cameras.values()]

    def get_camera(self, camera_id: str) -> Optional[CCTVCameraWorker]:
        return self.cameras.get(camera_id)
