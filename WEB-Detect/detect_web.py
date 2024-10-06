"""
Title: YOLOv8 Real-Time Object Detection using USB Camera (Web Version)
Author: ChatGPT 4o
Date: 10/6/2024

About:
This script demonstrates real-time object detection using the YOLOv8 model with a USB camera.
Instead of displaying the output using OpenCV windows, it hosts a web server to stream the 
detection results live, making it suitable for use on headless servers. The web server is built 
using Flask, allowing easy access from any device on the network.

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
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import logging
import os
import urllib.request

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

# Function to get color for a label based on hash value for consistency
def get_color(label):
    hash_value = hash(label) % 0xFFFFFF
    return (hash_value & 0xFF, (hash_value >> 8) & 0xFF, (hash_value >> 16) & 0xFF)

# Flask route to render the main page
@app.route('/')
def index():
    # HTML template for the web interface
    return render_template_string(f"""
    <!doctype html>
    <title>YOLOv8 Real-Time Object Detection</title>
    <h1>YOLOv8 Real-Time Object Detection Stream</h1>
    <h3>Using device: {device.upper()}</h3>
    <img src="/video_feed" width="100%">
    <p>Press Ctrl+C in the terminal to stop the server.</p>
    """)

# Video streaming generator function
def generate():
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

            # Get color for the label
            color = get_color(label)

            # Draw the bounding box outline
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare the label with confidence score
            label_text = f'{label}: {confidence:.2f}'

            # Determine label size
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            # Adjust label background position
            label_background_top_left = (x1, y1 - text_height - 10 if y1 - text_height - 10 > 10 else y1 + text_height + 10)
            label_background_bottom_right = (x1 + text_width, y1)

            # Draw a filled rectangle behind the text for better visibility
            cv2.rectangle(frame, label_background_top_left, label_background_bottom_right, color, -1)

            # Adjust label text position
            label_position = (x1, y1 - 5 if y1 - text_height - 10 > 10 else y1 + text_height + 5)

            # Put the label text on the frame
            cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode the frame for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Yield the frame in the correct format for a live stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask route to provide the video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the Flask server
if __name__ == '__main__':
    print("Starting Flask server... Access the video stream at http://localhost:5000")
    print(f"Using device: {device.upper()}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# Release resources when the script ends
cap.release()
