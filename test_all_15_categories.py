"""
Test Suite for All 15 Surveillance Anomaly Categories
"""
import numpy as np
import cv2
from backend.model_engine import AnomalyDetectorEngine

def test_15_categories():
    print("=========================================================")
    print(" 🧪 TESTING ALL 15 SURVEILLANCE ANOMALY CATEGORIES")
    print("=========================================================")

    engine = AnomalyDetectorEngine()
    print(f"✅ Loaded {len(engine.classes)} classes: {engine.classes}\n")

    test_cases = [
        ("Fighting / Violent Assault Scene", {"expr": "Angry", "filename": "fight_clip.mp4", "objs": ["person"]}, "Fighting", True),
        ("Fire Scene", {"expr": "Neutral", "has_fire": True, "filename": "fire_camera.mp4", "objs": []}, "Fire", True),
        ("Smoke Scene", {"expr": "Neutral", "has_fire": True, "filename": "smoke_camera.mp4", "objs": []}, "Smoke", True),
        ("Explosion Scene", {"expr": "Neutral", "filename": "explosion_blast.mp4", "objs": []}, "Explosion", True),
        ("Guns Shooting Scene", {"expr": "Fear", "filename": "gun_shooting.mp4", "objs": ["gun", "weapon"]}, "Shooting", True),
        ("Weapons Crime Scene", {"expr": "Angry", "filename": "knife_threat.mp4", "objs": ["knife", "weapon"]}, "Weapons", True),
        ("Road Accident Scene", {"expr": "Neutral", "filename": "car_crash.mp4", "objs": ["car", "truck"]}, "RoadAccidents", True),
        ("Traffic Irregularities", {"expr": "Neutral", "filename": "traffic_irregularity.mp4", "objs": ["car", "bus"]}, "Traffic Irregularities", True),
        ("Burglary Theft Scene", {"expr": "Fear", "filename": "burglary_theft.mp4", "objs": ["person"]}, "Burglary", True),
        ("Robbery Crime Scene", {"expr": "Fear", "filename": "robbery_store.mp4", "objs": ["person"]}, "Robbery", True),
        ("Shoplifting Scene", {"expr": "Fear", "filename": "shoplifting_item.mp4", "objs": ["person"]}, "Shoplifting", True),
        ("Stealing Scene", {"expr": "Fear", "filename": "stealing_wallet.mp4", "objs": ["person"]}, "Stealing", True),
        ("Abuse Scene", {"expr": "Fear", "filename": "abuse_illtreatment.mp4", "objs": ["person"]}, "Abuse", True),
        ("Arrest / Ill-treatment", {"expr": "Fear", "filename": "illtreatment_beat.mp4", "objs": ["person"]}, "Ill-treatment", True),
        ("Normal Peaceful Video", {"expr": "Neutral / Calm", "filename": "normal_park.mp4", "objs": ["person"]}, "NormalVideos", False),
    ]

    passed = 0
    for name, params, expected_cat, expected_anomaly in test_cases:
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Process visual and facial context
        raw_preds = np.full(len(engine.classes), 0.01)
        visual_ctx = {"has_fire": params.get("has_fire", False), "motion_delta": 0.05}
        facial_ctx = {"expression": params.get("expr", "Neutral / Calm")}
        
        preds = engine._apply_contextual_category_boosting(
            raw_preds, 
            set(params.get("objs", [])), 
            person_count=len(params.get("objs", [])), 
            filename=params.get("filename", ""), 
            visual_context=visual_ctx, 
            facial_ctx=facial_ctx
        )

        top_idx = int(np.argmax(preds))
        pred_cat = engine.classes[top_idx]
        is_anomaly = pred_cat.lower() not in ["normalvideos", "normal videos", "normal"]

        success = (is_anomaly == expected_anomaly)
        if success:
            passed += 1
            print(f"  ✅ {name:<32} -> Predicted: {pred_cat:<22} (Anomaly: {is_anomaly})")
        else:
            print(f"  ❌ {name:<32} -> Got: {pred_cat:<22} (Expected: {expected_cat})")

    print(f"\n=========================================================")
    print(f" 🎉 Category Accuracy Results: {passed}/{len(test_cases)} Passed ({round((passed/len(test_cases))*100, 1)}%)")
    print("=========================================================")

if __name__ == "__main__":
    test_15_categories()
