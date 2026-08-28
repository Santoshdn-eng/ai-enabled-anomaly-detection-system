"""
AI Surveillance Anomaly Detection System - Desktop Application Launcher.
Launches the FastAPI backend server in a background thread and opens a native desktop GUI window.
"""

import sys
import os
import time
import threading
import webbrowser
import uvicorn

def run_backend():
    """Runs FastAPI server on port 8000"""
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, log_level="error")

def main():
    print("=" * 60)
    print("🛡️  A.I. Surveillance Anomaly Detection System Desktop App Launcher")
    print("=" * 60)
    
    # 1. Start FastAPI server in daemon thread
    server_thread = threading.Thread(target=run_backend, daemon=True)
    server_thread.start()
    
    app_url = "http://127.0.0.1:8000"
    print(f"⚡ Starting local server at {app_url}...")
    time.sleep(1.5)

    # 2. Try launching via PyWebView native desktop window, fallback to browser
    try:
        import webview
        print("🖥️ Opening Desktop Native GUI Window...")
        webview.create_window(
            title="A.I. Surveillance Anomaly Detector & CCTV Monitor",
            url=app_url,
            width=1400,
            height=900,
            resizable=True
        )
        webview.start()
    except Exception as e:
        print(f"ℹ️ PyWebView native GUI fallback ({e}). Opening in system default web browser...")
        webbrowser.open(app_url)
        print("🟢 Application running! Press Ctrl+C in terminal to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")

if __name__ == "__main__":
    main()
