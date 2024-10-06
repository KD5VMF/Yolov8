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
from flask import Flask, Response, render_template_string, request, redirect, url_for
from ultralytics import YOLO
import logging
import os
import urllib.request
import threading

try:
    from screeninfo import get_monitors
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height
except (ImportError, Exception):
    # Default resolution for headless servers
    screen_width, screen_height = 1280, 720
    print("No monitor detected. Using default resolution of 1280x720.")

# Hide FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress additional YOLO logging output
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Function to download the YOLOv8 model weights if not present
def download_weights(model_name, url):
    if not os.path.isfile(model_name):
        print(f"Downloading weights file '{model_name}' from {url}...")
        urllib.request.urlretrieve(url, model_name)
        print(f"Download complete: '{model_name}'")

# Download YOLOv8 weights if not present
model_name = "yolov8x.pt"
model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
download_weights(model_name, model_url)

# Load the YOLOv8 model using the Ultralytics YOLO library
print(f"Loading YOLOv8 model '{model_name}'...")
model = YOLO(model_name)

# Set device to CUDA if available, else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using device: {device.upper()}")

# Flask app setup
app = Flask(__name__)

# Initialize USB camera (index 0) with V4L2 backend, suitable for Linux
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream from USB camera.")
    exit()

# Selected classes for detection (initially set to None)
selected_classes = None

# Function to get color for a label based on hash value for consistency
def get_color(label):
    hash_value = hash(label) % 0xFFFFFF
    return (hash_value & 0xFF, (hash_value >> 8) & 0xFF, (hash_value >> 16) & 0xFF)

# Flask route to render the selection page
@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_classes
    if request.method == 'POST':
        user_input = request.form.get('classes')
        if user_input.lower() == 'all':
            selected_classes = list(model.names.values())
        else:
            try:
                class_indices = [int(idx) - 1 for idx in user_input.split(",")]
                selected_classes = [model.names[idx] for idx in class_indices if 0 <= idx < len(model.names)]
            except ValueError:
                selected_classes = list(model.names.values())

        return redirect(url_for('video_feed'))
    
    # Render selection template
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

    return render_template_string(f"""
    <!doctype html>
    <title>YOLOv8 Real-Time Object Detection</title>
    <h1>Select Classes for Detection</h1>
    <form method="post">
        <div>
            {class_list_html}
        </div>
        <p>Enter the numbers of the classes you want to detect (comma-separated) or type 'ALL' for all classes:</p>
        <input type="text" name="classes" required>
        <button type="submit">Start Detection</button>
    </form>
    """)

# Flask route to provide the video stream
@app.route('/video_feed')
def video_feed():
    global selected_classes
    if selected_classes is None:
        # Redirect to the class selection page if no classes have been selected
        return redirect(url_for('index'))
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Video streaming generator function
def generate():
    global cap, selected_classes, screen_width, screen_height
    while True:
        # Capture frame from USB camera
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (YOLO expects RGB format)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model(img_rgb, verbose=False)

        # Extract detection data (labels, confidence scores, and bounding boxes)
        detections = results[0].boxes

        # Render results on the original frame
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            label = model.names[int(detection.cls)]
            confidence = detection.conf[0].item()

            # Only render if the label is in the selected classes
            if selected_classes and label not in selected_classes:
                continue

            # Get color for the label
            color = get_color(label)

            # Draw the bounding box outline
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare the label with confidence score
            label_text = f'{label}: {confidence:.2f}'

            # Determine label size
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Adjust label background position
            label_background_top_left = (x1, y1 - text_height - 10 if y1 - text_height - 10 > 10 else y1 + text_height + 10)
            label_background_bottom_right = (x1 + text_width, y1)

            # Draw a filled rectangle behind the text for better visibility
            cv2.rectangle(frame, label_background_top_left, label_background_bottom_right, color, -1)

            # Adjust label text position
            label_position = (x1, y1 - 5 if y1 - text_height - 10 > 10 else y1 + text_height + 5)

            # Put the label text on the frame
            cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add text overlay at the top and bottom
        overlay_text_top = f"YOLOv8 Real-Time Detection - Device: {device.upper()}"
        overlay_text_bottom = f"Using Model: {model_name} - Powered by ChatGPT 4o"

        # Add black background for the text
        frame_height, frame_width, _ = frame.shape
        top_rect_height = 40
        bottom_rect_height = 40
        cv2.rectangle(frame, (0, 0), (frame_width, top_rect_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, frame_height - bottom_rect_height), (frame_width, frame_height), (0, 0, 0), -1)

        # Add text to the top and bottom of the frame
        cv2.putText(frame, overlay_text_top, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, overlay_text_bottom, (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Resize the frame to fit the screen dimensions
        resized_frame = cv2.resize(frame, (screen_width, screen_height))

        # Encode the frame for streaming
        _, jpeg = cv2.imencode('.jpg', resized_frame)
        frame_bytes = jpeg.tobytes()

        # Yield the frame in the correct format for a live stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask route to stop the server
@app.route('/stop')
def stop():
    shutdown_server()
    return "Server shutting down..."

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()

# Start the Flask server
if __name__ == '__main__':
    print("Starting Flask server... Access the video stream at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# Release resources when the script ends
cap.release()
