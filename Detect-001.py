"""
Title: YOLOv8 Real-Time Object Detection using USB Camera
Author: ChatGTP 4o
Date: 10/5/2024

About:
This script demonstrates real-time object detection using the YOLOv8 model with a USB camera.
YOLOv8, developed by Ultralytics, is the latest version offering high accuracy and performance
for a variety of object detection tasks. The script automatically detects whether a GPU is available
for acceleration and downloads the YOLOv8 model if it is not already present. Users can choose
which objects to detect by selecting from a list of available classes.

Requirements:
- Python 3.8 or above
- OpenCV
- PyTorch
- Ultralytics YOLO library
"""

import warnings
import torch
import cv2
import numpy as np
from screeninfo import get_monitors
from ultralytics import YOLO
import logging
import os
import urllib.request
import platform

# Hide FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress additional YOLO logging output
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Function to clear the console screen
def clear_screen():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

# Clear the screen at the start of the script
clear_screen()

# Title and About information printed when the script runs
print("""
====================================================
YOLOv8 Real-Time Object Detection using USB Camera
====================================================
Author: CHAT-GPT 4o
Date: 10/5/2024

About:
This script demonstrates real-time object detection using the YOLOv8 model with a USB camera.
YOLOv8 is the latest model developed by Ultralytics, known for its high accuracy and performance.

====================================================
""")

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
print(f"Using device: {device}")

# Display list of available classes in a neat aligned format and get user input
available_classes = list(model.names.values())  # Convert class names to a list
print("\nAvailable classes for detection:")

# Format and print the available classes in multiple columns for better readability
columns = 4
for i in range(0, len(available_classes), columns):
    row = ""
    for j in range(columns):
        if i + j < len(available_classes):
            row += f"{i + j + 1:2}. {available_classes[i + j]:<15}"
    print(row)

user_input = input("\nEnter the numbers of the classes you want to detect (comma-separated) or type 'ALL' for all classes: ").strip()

if user_input.lower() == 'all':
    selected_classes = available_classes
else:
    try:
        class_indices = [int(idx) - 1 for idx in user_input.split(",")]
        selected_classes = [available_classes[idx] for idx in class_indices if 0 <= idx < len(available_classes)]
    except ValueError:
        print("Invalid input. Defaulting to all classes.")
        selected_classes = available_classes

print(f"\nDetecting the following classes: {', '.join(selected_classes)}")

# Function to get color for a label
def get_color(label):
    color_map = {
        'person': (0, 255, 0),
        'car': (0, 0, 255),
        'bicycle': (255, 0, 0),
    }
    if label in color_map:
        return color_map[label]
    else:
        hash_value = hash(label) % 0xFFFFFF
        return (hash_value & 0xFF, (hash_value >> 8) & 0xFF, (hash_value >> 16) & 0xFF)

# Initialize USB camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from USB camera.")
    exit()

# Get screen resolution dynamically
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Create a named window for the display and maximize it
cv2.namedWindow('YOLOv8 Object Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('YOLOv8 Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Starting object detection... Press 'q' to quit.")

# Main loop for real-time object detection
while True:
    # Capture frame from USB camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
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

        # Only display selected classes
        if label not in selected_classes:
            continue

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

    # Resize the annotated frame to fit the screen dimensions
    resized_frame = cv2.resize(frame, (screen_width, screen_height))

    # Display the output
    cv2.imshow('YOLOv8 Object Detection', resized_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Object detection ended. Goodbye!")
