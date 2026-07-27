"""
Generate sample surveillance video clips (.mp4) for testing the AI Anomaly Detection System.
"""

import os
import cv2
import numpy as np

def create_sample_videos():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_samples")
    os.makedirs(output_dir, exist_ok=True)

    fps = 25
    num_frames = 50  # 2 seconds
    height, width = 256, 256

    # 1. Normal Video Clip
    normal_path = os.path.join(output_dir, "sample_normal_surveillance.mp4")
    # pyrefly: ignore [missing-attribute]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_normal = cv2.VideoWriter(normal_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), (30, 30, 40), dtype=np.uint8)
        # Smooth motion
        cx = int(50 + (i * 3) % 150)
        cy = 128
        cv2.circle(frame, (cx, cy), 15, (0, 200, 100), -1)
        cv2.putText(frame, "NORMAL SURVEILLANCE FEED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        out_normal.write(frame)
    out_normal.release()

    # 2. Anomaly Video Clip (Rapid Motion / Fight Simulation)
    anomaly_path = os.path.join(output_dir, "sample_fighting_anomaly.mp4")
    out_anomaly = cv2.VideoWriter(anomaly_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), (20, 20, 35), dtype=np.uint8)
        # Eratic motion simulation
        cx1 = int(128 + np.sin(i * 0.8) * 45)
        cy1 = int(128 + np.cos(i * 0.8) * 35)
        cx2 = int(128 - np.sin(i * 0.8) * 45)
        cy2 = int(128 - np.cos(i * 0.8) * 35)

        cv2.circle(frame, (cx1, cy1), 20, (50, 50, 230), -1)
        cv2.circle(frame, (cx2, cy2), 20, (50, 220, 230), -1)
        if i % 4 == 0:
            cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)

        cv2.putText(frame, "ANOMALY FEED SIMULATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        out_anomaly.write(frame)
    out_anomaly.release()

    print("✅ Sample test videos created successfully in data/custom_samples/:")
    print(f"  1. Normal:  {normal_path}")
    print(f"  2. Anomaly: {anomaly_path}")

if __name__ == "__main__":
    create_sample_videos()
