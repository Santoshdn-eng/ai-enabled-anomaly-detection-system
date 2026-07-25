import os
import html
import subprocess
import datetime

PROJECT_ROOT = "/Users/santoshdebnath/Desktop/AI Enabled Anomaly Detection System"
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "AI_Enabled_Anomaly_Detection_System_Complete_Codebase_and_Report.pdf")

FILES_TO_INCLUDE = [
    ("README.md", "Project Overview & Documentation", "markdown"),
    ("requirements.txt", "Project Dependencies", "text"),
    ("backend/main.py", "FastAPI Backend API Server & WebSockets", "python"),
    ("backend/model_engine.py", "Deep Learning Inference & Spatial Fusion Boosting Engine", "python"),
    ("model/train_improved.py", "Model Training, Data Augmentation & Serialization Pipeline", "python"),
    ("hf_integration/pull_from_hf.py", "Hugging Face Model Pull Utility", "python"),
    ("hf_integration/push_to_hf.py", "Hugging Face Model Upload & Versioning", "python"),
    ("data/create_sample_videos.py", "Synthetic Video Generator for Testing", "python"),
    ("data/download_yolo_dataset.py", "YOLO Dataset Download & Preparation", "python"),
    ("frontend/index.html", "Frontend Web Dashboard HTML Structure", "html"),
    ("frontend/style.css", "Frontend Styling & Glassmorphism Design", "css"),
    ("frontend/app.js", "Frontend Logic & WebSocket Streaming Client", "javascript"),
]

def get_file_content(relative_path):
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return f"/* File {relative_path} not found */"

def generate_html():
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Enabled Anomaly Detection System - Project Report & Codebase</title>
<style>
    @page {{
        size: A4;
        margin: 20mm 15mm 20mm 15mm;
    }}
    body {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
        color: #1a1a2e;
        line-height: 1.5;
        font-size: 11pt;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }}
    
    .cover-page {{
        height: 90vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        page-break-after: always;
        border: 2px solid #3b82f6;
        padding: 40px;
        border-radius: 12px;
        margin-top: 20px;
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
    }}
    
    .cover-title {{
        font-size: 28pt;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 10px;
        line-height: 1.2;
    }}
    
    .cover-subtitle {{
        font-size: 16pt;
        color: #3b82f6;
        margin-bottom: 40px;
        font-weight: 600;
    }}
    
    .cover-meta {{
        font-size: 12pt;
        color: #475569;
        margin-top: 60px;
        line-height: 1.8;
    }}
    
    .badge {{
        display: inline-block;
        background-color: #2563eb;
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 10pt;
        font-weight: 600;
        margin-bottom: 25px;
    }}
    
    h1 {{
        font-size: 20pt;
        color: #1e3a8a;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 6px;
        margin-top: 30px;
        page-break-before: always;
    }}
    
    h2 {{
        font-size: 14pt;
        color: #1e40af;
        margin-top: 20px;
    }}
    
    h3 {{
        font-size: 12pt;
        color: #0f766e;
    }}

    .section-block {{
        margin-bottom: 25px;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 10pt;
    }}

    th, td {{
        border: 1px solid #cbd5e1;
        padding: 8px 12px;
        text-align: left;
    }}

    th {{
        background-color: #1e293b;
        color: #ffffff;
        font-weight: 600;
    }}

    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}

    .info-box {{
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 12px 16px;
        margin: 15px 0;
        border-radius: 4px;
    }}

    .model-card {{
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }}

    .file-header {{
        background-color: #0f172a;
        color: #38bdf8;
        padding: 10px 15px;
        font-family: 'Courier New', monospace;
        font-size: 11pt;
        font-weight: bold;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-top: 25px;
        display: flex;
        justify-content: space-between;
        page-break-inside: avoid;
    }}

    .file-desc {{
        color: #94a3b8;
        font-size: 9pt;
        font-weight: normal;
    }}

    pre {{
        background-color: #020617;
        color: #f1f5f9;
        padding: 14px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 8.5pt;
        line-height: 1.4;
        border-bottom-left-radius: 6px;
        border-bottom-right-radius: 6px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-all;
        margin-top: 0;
        margin-bottom: 25px;
        border: 1px solid #1e293b;
    }}

    .toc-item {{
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px dashed #cbd5e1;
    }}
</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover-page">
    <div class="badge">PROJECT COMPREHENSIVE DOCUMENTATION & CODEBASE</div>
    <div class="cover-title">AI Enabled Anomaly Detection System</div>
    <div class="cover-subtitle">Surveillance Anomaly Detection using Deep Learning, Transformers & YOLO Spatial Boosting</div>
    
    <div class="cover-meta">
        <strong>Author / Developer:</strong> Santosh Debnath<br>
        <strong>Hugging Face Repository:</strong> <code>SantoshDN/ai-enable-anomaly-detection</code><br>
        <strong>Core Frameworks:</strong> TensorFlow / Keras, FastAPI, OpenCV, YOLOv8<br>
        <strong>Generated Date:</strong> {current_date}
    </div>
</div>

<!-- EXECUTIVE SUMMARY & ARCHITECTURE -->
<h1>1. Executive Project & Architecture Overview</h1>
<div class="section-block">
    <p>This report documents the complete source code, trained model configurations, and architecture specifications for the <strong>AI Enabled Anomaly Detection System</strong>. The system provides real-time surveillance video analysis for identifying anomalies such as fighting, violence, burglaries, fires, traffic accidents, and unauthorized intrusion.</p>

    <h2>1.1 System Architecture & Technical Pipeline</h2>
    <ul>
        <li><strong>Feature Extractor Backbone:</strong> MobileNetV2 / EfficientNetB0 pre-trained on ImageNet to extract spatial feature maps from video frames (resolution: 128x128).</li>
        <li><strong>Temporal Sequence Modeling:</strong> <code>TimeDistributed</code> layer wrapper paired with a <code>Bidirectional LSTM</code> network to model temporal motion across a sequence of 16 consecutive frames.</li>
        <li><strong>Custom Positional Embedding:</strong> Adds sequence ordering awareness across temporal frame batches.</li>
        <li><strong>Spatial Object & Person Boosting (YOLOv8):</strong> Integrated YOLOv8n / YOLOv8-pose to detect human presence, person count, and context weapons/tools to boost anomaly confidence dynamically.</li>
        <li><strong>Backend Service:</strong> FastAPI web server providing REST endpoints (<code>/api/predict</code>, <code>/api/predict_frame</code>, <code>/api/health</code>) and high-performance WebSockets (<code>/ws/live</code>).</li>
        <li><strong>Frontend Client:</strong> Lightweight responsive Dashboard with real-time video streaming, risk timeline graphs, and live confidence metrics.</li>
    </ul>

    <h2>1.2 Trained Models & Weights Summary</h2>
    <p>The following trained model files were developed and integrated into the system:</p>
    <table>
        <thead>
            <tr>
                <th>Model File / Artifact</th>
                <th>Model Architecture</th>
                <th>Purpose / Description</th>
                <th>Storage / Deployment Location</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>best_model.keras</code> / <code>models/best_model.keras</code></td>
                <td>EfficientNet / MobileNet + TimeDistributed + LSTM</td>
                <td>Primary Anomaly Classifier trained on multi-class video datasets. Serialized with custom <code>PositionalEmbedding</code> layer.</td>
                <td>Local Directory & Hugging Face Hub (<code>SantoshDN/ai-enable-anomaly-detection</code>)</td>
            </tr>
            <tr>
                <td><code>yolov8n.pt</code></td>
                <td>YOLOv8 Nano Neural Network</td>
                <td>Real-time spatial object detection (person, weapons, vehicles).</td>
                <td>Local Directory (PyTorch format)</td>
            </tr>
            <tr>
                <td><code>yolov8n-pose.pt</code></td>
                <td>YOLOv8 Pose Estimation Network</td>
                <td>Human keypoint & stance detection for anomaly risk analysis.</td>
                <td>Local Directory (PyTorch format)</td>
            </tr>
        </tbody>
    </table>

    <div class="info-box">
        <strong>Hugging Face Integration:</strong> Trained models are automatically synchronized and pulled from Hugging Face Hub repository: <code>https://huggingface.co/SantoshDN/ai-enable-anomaly-detection</code> using <code>pull_from_hf.py</code> and <code>push_to_hf.py</code>.
    </div>

    <h2>1.3 Model Hyperparameters & Training Configuration</h2>
    <table>
        <tr><th>Frame Resolution</th><td>128 x 128 pixels (3 channels RGB)</td></tr>
        <tr><th>Sequence Length</th><td>16 frames per clip window</td></tr>
        <tr><th>Batch Size</th><td>8 / 16 video sequences</td></tr>
        <tr><th>Optimizer</th><td>Adam (Learning Rate = 1e-4 with ReduceLROnPlateau)</td></tr>
        <tr><th>Loss Function</th><td>Categorical Crossentropy</td></tr>
        <tr><th>Target Classes</th><td>Fighting, Violence, Arrest, Robbery, Burglary, Abuse, Normal Videos, etc.</td></tr>
    </table>
</div>

<!-- TABLE OF CONTENTS -->
<h1>2. Project File Structure & Codebase Directory</h1>
<div class="section-block">
    <p>The codebase is organized cleanly into modular backend, frontend, model training, data generation, and Hugging Face integration scripts:</p>
"""

    for rel_path, desc, lang in FILES_TO_INCLUDE:
        html_out += f"""
    <div class="toc-item">
        <span><strong>{rel_path}</strong> - {desc}</span>
        <span>[{lang.upper()}]</span>
    </div>"""

    html_out += """
</div>

<!-- COMPLETE SOURCE CODE SECTION -->
<h1>3. Complete Project Source Code</h1>
"""

    for idx, (rel_path, desc, lang) in enumerate(FILES_TO_INCLUDE, 1):
        content = get_file_content(rel_path)
        escaped_code = html.escape(content)
        
        html_out += f"""
<div class="file-block">
    <div class="file-header">
        <span>3.{idx} {rel_path}</span>
        <span class="file-desc">{desc}</span>
    </div>
    <pre><code>{escaped_code}</code></pre>
</div>
"""

    html_out += """
</body>
</html>
"""
    return html_out

def main():
    print("Building HTML document for PDF conversion...")
    html_content = generate_html()
    
    html_path = os.path.join(PROJECT_ROOT, "codebase_report_temp.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"HTML rendered successfully to {html_path}")
    print("Converting HTML to PDF via Chrome Headless...")
    
    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless=new",
        f"--print-to-pdf={OUTPUT_PDF}",
        "--no-pdf-header-footer",
        html_path
    ]
    
    result = subprocess.run(chrome_cmd, capture_output=True, text=True)
    if os.path.exists(OUTPUT_PDF):
        size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
        print(f"SUCCESS! PDF created at: {OUTPUT_PDF} ({size_mb:.2f} MB)")
    else:
        print(f"Error generating PDF: {result.stderr}")

if __name__ == "__main__":
    main()
