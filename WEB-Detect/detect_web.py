"""
Title: YOLOv8 Real-Time Object Detection using USB Camera (Web Version)
Author: ChatGPT 4o
Date: 10/6/2024

About:
This script demonstrates real-time object detection using the YOLOv8 model with a USB camera.
Instead of displaying the output using OpenCV windows, it hosts a web server to stream the 
detection results live, making it suitable for use on headless servers. The web server is built 
using Flask, allowing easy access from any device on the network.

Users can select which object classes to detect, and there is a mechanism to stop the video stream
either by the user or by detecting if the viewer has left.

Requirements:
- Python 3.8 or above
- OpenCV
- PyTorch
- Ultralytics YOLO library
- Flask
- CUDA Toolkit (for GPU acceleration)
"""

import warnings
import torch
import cv2
import numpy as np
import time
from flask import Flask, Response, render_template_string, request, redirect, url_for
from ultralytics import YOLO
import logging
import os
import urllib.request

try:
    from screeninfo import get_monitors
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height
except (ImportError, Exception):
    screen_width, screen_height = 1280, 720
    print("No monitor detected. Using default resolution of 1280x720.")

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Function to download YOLOv8 model weights if not present
def download_weights(model_name, url):
    if not os.path.isfile(model_name):
        print(f"Downloading weights file '{model_name}' from {url}...")
        urllib.request.urlretrieve(url, model_name)
        print(f"Download complete: '{model_name}'")

# Default model (initial model before user selection)
model_name = "yolov8x.pt"
model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
download_weights(model_name, model_url)

# Initialize Flask app
app = Flask(__name__)

# Initialize USB camera (index 0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open video stream from USB camera.")
    exit()

selected_classes = None
selected_device = None  # To track CPU/GPU selection
frame_skip = 1  # Variable to handle automatic frame skipping

def get_color(label):
    """Get color for label based on hash value."""
    hash_value = hash(label) % 0xFFFFFF
    return (hash_value & 0xFF, (hash_value >> 8) & 0xFF, (hash_value >> 16) & 0xFF)

@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_classes, selected_device, model_name
    
    # Ensure model is initialized here
    model = YOLO(model_name)  # Load the default model (YOLOv8x)
    
    if request.method == 'POST':
        user_input = request.form.get('classes')
        device_input = request.form.get('device')
        model_input = request.form.get('model')  # Add model selection

        # Parse classes
        if user_input.lower() == 'all':
            selected_classes = list(model.names.values())
        else:
            try:
                class_indices = [int(idx) - 1 for idx in user_input.split(",")]
                selected_classes = [model.names[idx] for idx in class_indices if 0 <= idx < len(model.names)]
            except ValueError:
                selected_classes = list(model.names.values())

        selected_device = device_input
        model_name = f"yolov8{model_input}.pt"  # Update model based on user selection
        return redirect(url_for('video_feed'))

    # Display available classes for selection
    available_classes = list(model.names.values())
    class_list_html = "<table style='width: 100%;'>"
    columns = 4
    for i in range(0, len(available_classes), columns):
        class_list_html += "<tr>"
        for j in range(columns):
            if i + j < len(available_classes):
                class_index = i + j + 1
                class_name = available_classes[i + j]
                class_list_html += f"<td style='padding: 10px;'>{class_index:2}. {class_name:<15}</td>"
        class_list_html += "</tr>"
    class_list_html += "</table>"

    # Model descriptions
    model_descriptions = {
        'n': 'YOLOv8n (Nano) - Fastest, lowest accuracy',
        's': 'YOLOv8s (Small) - Balance of speed and accuracy',
        'm': 'YOLOv8m (Medium) - Moderate speed, higher accuracy',
        'l': 'YOLOv8l (Large) - Slower, better accuracy',
        'x': 'YOLOv8x (Extra Large) - Slowest, highest accuracy'
    }

    # Generate the dropdown with model descriptions
    model_options_html = ''.join([f'<option value="{model}">{model_descriptions[model]}</option>' for model in model_descriptions])

    return render_template_string(f"""
    <!doctype html>
    <title>YOLOv8 Real-Time Object Detection</title>
    <h1>Select Classes for Detection and Device</h1>
    <form method="post">
        <div>{class_list_html}</div>
        <p>Enter the numbers of the classes you want to detect (comma-separated) or type 'ALL' for all classes:</p>
        <input type="text" name="classes" required>
        
        <h2>Select Device for Detection</h2>
        <label><input type="radio" name="device" value="cpu" checked> CPU</label><br>
        <label><input type="radio" name="device" value="cuda"> GPU (if available)</label><br>
        
        <h2>Select Model Variant</h2>
        <select name="model">
            {model_options_html}
        </select><br><br>

        <button type="submit">Start Detection</button>
    </form>
    """)

@app.route('/video_feed')
def video_feed():
    global selected_classes, selected_device
    if selected_classes is None or selected_device is None:
        return redirect(url_for('index'))
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    global cap, selected_classes, selected_device, screen_width, screen_height, frame_skip

    # Load model on the selected device
    print(f"Loading YOLOv8 model '{model_name}' on {selected_device.upper()}...")
    device = selected_device if torch.cuda.is_available() and selected_device == 'cuda' else 'cpu'
    model = YOLO(model_name).to(device)
    print(f"Using device: {device.upper()}")

    frame_count = 0  # Counter for skipping frames

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Skip frames if frame_skip > 1 (only process every nth frame)
        if frame_count % frame_skip != 0:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, verbose=False)
        detections = results[0].boxes

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            label = model.names[int(detection.cls)]
            confidence = detection.conf[0].item()

            if selected_classes and label not in selected_classes:
                continue

            color = get_color(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f'{label}: {confidence:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            label_background_top_left = (x1, y1 - text_height - 10 if y1 - text_height - 10 > 10 else y1 + text_height + 10)
            label_background_bottom_right = (x1 + text_width, y1)

            cv2.rectangle(frame, label_background_top_left, label_background_bottom_right, color, -1)
            label_position = (x1, y1 - 5 if y1 - text_height - 10 > 10 else y1 + text_height + 5)
            cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)

        # Overlay FPS and device info on the frame
        overlay_text_top = f"YOLOv8 Real-Time Detection - Device: {device.upper()} - FPS: {fps:.2f}"
        overlay_text_bottom = f"Using Model: {model_name} - Powered by ChatGPT 4o"
        frame_height, frame_width, _ = frame.shape
        top_rect_height = 40
        bottom_rect_height = 40

        # Add black background for the text
        cv2.rectangle(frame, (0, 0), (frame_width, top_rect_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, frame_height - bottom_rect_height), (frame_width, frame_height), (0, 0, 0), -1)

        cv2.putText(frame, overlay_text_top, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, overlay_text_bottom, (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        resized_frame = cv2.resize(frame, (screen_width, screen_height))
        _, jpeg = cv2.imencode('.jpg', resized_frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/stop')
def stop():
    shutdown_server()
    return "Server shutting down..."

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()

if __name__ == '__main__':
    print("Starting Flask server... Access the video stream at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

cap.release()
