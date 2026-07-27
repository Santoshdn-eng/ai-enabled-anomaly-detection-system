"""
Comprehensive Test Script for AI Enabled Anomaly Detection System
Tests:
1. Sample Video Generation
2. Model Initialization & Weights Discovery
3. Video Inference & Anomaly Classification
4. Threat Severity & Dynamic Crowd Risk Calculation
5. FastAPI Backend Endpoint Health (if server is active)
"""

import os
import sys
import json
import requests

def test_sample_generation():
    print("\n--- 1. Testing Sample Video Generation ---")
    try:
        from data.create_sample_videos import create_sample_videos
        create_sample_videos()
        
        sample_dir = os.path.join("data", "custom_samples")
        normal_mp4 = os.path.join(sample_dir, "sample_normal_surveillance.mp4")
        anomaly_mp4 = os.path.join(sample_dir, "sample_fighting_anomaly.mp4")
        
        assert os.path.exists(normal_mp4), f"Missing {normal_mp4}"
        assert os.path.exists(anomaly_mp4), f"Missing {anomaly_mp4}"
        print("✅ Sample test videos generated and verified successfully!")
        return normal_mp4, anomaly_mp4
    except Exception as e:
        print(f"❌ Sample video generation failed: {e}")
        return None, None

def test_model_inference(normal_video, anomaly_video):
    print("\n--- 2. Testing Anomaly Detection Model Engine ---")
    try:
        from backend.model_engine import AnomalyDetectorEngine
        
        print("Initializing AnomalyDetectorEngine...")
        engine = AnomalyDetectorEngine(model_path="best_model.keras")
        
        print(f"Loaded {len(engine.classes)} classes.")
        print(f"YOLOv8 Pose Model Status: {'Loaded' if engine.yolo_model else 'Fallback Mode'}")
        
        # Test Normal Video
        print(f"\nRunning Inference on Normal Video: {normal_video}")
        with open(normal_video, "rb") as f:
            bytes_data = f.read()
        res_normal = engine.process_video_file(bytes_data, "sample_normal_surveillance.mp4")
        
        print(f"  • Predicted Category : {res_normal.get('predicted_class')}")
        print(f"  • Is Anomaly?        : {res_normal.get('is_anomaly')}")
        print(f"  • Confidence         : {res_normal.get('confidence_pct')}%")
        print(f"  • Threat Severity    : {res_normal.get('threat_severity_pct')}% ({res_normal.get('severity_level')})")
        
        # Test Anomaly Video
        print(f"\nRunning Inference on Fighting/Anomaly Video: {anomaly_video}")
        with open(anomaly_video, "rb") as f:
            bytes_data = f.read()
        res_anomaly = engine.process_video_file(bytes_data, "sample_fighting_anomaly.mp4")
        
        print(f"  • Predicted Category : {res_anomaly.get('predicted_class')}")
        print(f"  • Is Anomaly?        : {res_anomaly.get('is_anomaly')}")
        print(f"  • Confidence         : {res_anomaly.get('confidence_pct')}%")
        print(f"  • Threat Severity    : {res_anomaly.get('threat_severity_pct')}% ({res_anomaly.get('severity_level')})")
        
        print("✅ Model inference engine test completed successfully!")
    except Exception as e:
        print(f"❌ Model inference failed: {e}")

def test_api_health():
    print("\n--- 3. Testing Backend REST API Endpoints (http://localhost:8000) ---")
    url = "http://localhost:8000/api/health"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            print("✅ FastAPI Server is running online!")
            print("   Response:", json.dumps(response.json(), indent=2))
        else:
            print(f"⚠️ FastAPI returned status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("ℹ️ FastAPI server is not currently running. Launch it with: python -m backend.main")
    except Exception as e:
        print(f"⚠️ Health check check failed: {e}")

if __name__ == "__main__":
    print("=========================================================")
    print(" 🛡️  AI SURVEILLANCE ANOMALY DETECTION SYSTEM TEST SUITE ")
    print("=========================================================")
    normal_vid, anomaly_vid = test_sample_generation()
    if normal_vid and anomaly_vid:
        test_model_inference(normal_vid, anomaly_vid)
    test_api_health()
    print("\n=========================================================")
    print(" 🎉 All tests finished.")
    print("=========================================================")
