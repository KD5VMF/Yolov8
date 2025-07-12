"""
Title: YOLOv8 Real-Time Object Detection using USB Camera with GPU Selection
Author: ChatGTP 4o
Date: 10/5/2024

About:
This script demonstrates real-time object detection using the YOLOv8 model with a USB camera.
It now allows the user to select which GPU to use if multiple GPUs are available.
YOLOv8, developed by Ultralytics, is the latest version offering high accuracy and performance
for a variety of object detection tasks. The script automatically detects whether a GPU is available
for acceleration and downloads the YOLOv8 model if it is not already present. Users can choose
which objects to detect by selecting from a list of available classes.
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

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def clear_screen():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def ascii_header():
    print(r"""
 __     __   ____   _       ____   __     __  ____    ___  
 \ \   / /  / __ \ | |     / __ \  \ \   / / / __ \  / _ \ 
  \ \_/ /  | |  | || |    | |  | |  \ \_/ / | |  | || | | |
   \   /   | |  | || |    | |  | |   \   /  | |  | || | | |
    | |    | |__| || |____| |__| |    | |   | |__| || |_| |
    |_|     \____/ |______|\____/     |_|    \____/  \___/ 
                                                            
                Real-Time Object Detection                 
               Powered by YOLOv8 - Ultralytics              
============================================================
""")

def download_weights(model_name, url):
    if not os.path.isfile(model_name):
        print(f"\n>>> Downloading weights '{model_name}'...")
        urllib.request.urlretrieve(url, model_name)
        print(f">>> Download complete: '{model_name}'")

def select_model():
    print("\n[ SELECT YOLOv8 MODEL ]")
    print("┌────┬────────────┬────────────────────────────────────────────┐")
    print("│Key │ Model      │ Description                                │")
    print("├────┼────────────┼────────────────────────────────────────────┤")
    print("│ N  │ yolov8n.pt │ Nano    (Fastest, lowest accuracy)         │")
    print("│ S  │ yolov8s.pt │ Small   (Very fast, lower accuracy)        │")
    print("│ M  │ yolov8m.pt │ Medium  (Balanced speed/accuracy)          │")
    print("│ L  │ yolov8l.pt │ Large   (High accuracy, slower)            │")
    print("│ X  │ yolov8x.pt │ XLarge  (Highest accuracy, slowest)        │")
    print("└────┴────────────┴────────────────────────────────────────────┘")
    choice = input("\nSelect model (n/s/m/l/x) [default x]: ").strip().lower()
    models = {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt", "l": "yolov8l.pt", "x": "yolov8x.pt"}
    model_key = choice if choice in models else "x"
    print(f">>> Selected model: {models[model_key]}")
    return models[model_key]

def select_device():
    print("\n[ SELECT DEVICE ]")
    print("┌────┬─────────────────────────────────────────────┬─────────────────────────────────┐")
    print("│ No │ Device                                      │ Notes                           │")
    print("├────┼─────────────────────────────────────────────┼─────────────────────────────────┤")
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            trimmed_name = name[:39] if len(name) > 39 else name
            devices.append((f"cuda:{i}", name))
            print(f"│ {i:<2} │ GPU {trimmed_name:<39} │ Best performance, recommended   │")
    cpu_index = len(devices)
    devices.append(("cpu", "CPU"))
    print(f"│ {cpu_index:<2} │ CPU                                         │ Slower, but universal           │")
    print("└────┴─────────────────────────────────────────────┴─────────────────────────────────┘")

    try:
        choice = input(f"\nSelect device by number [default 0]: ").strip()
        index = int(choice) if choice else 0
        if index < 0 or index >= len(devices):
            print("Invalid choice. Using default (0).")
            index = 0
    except:
        print("Invalid input. Using default (0).")
        index = 0

    print(f">>> Selected device: {devices[index][1]}")
    return devices[index][0]

def select_classes(available_classes):
    print("\n[ SELECT CLASSES TO DETECT ]")
    columns = 4
    for i in range(0, len(available_classes), columns):
        print('  '.join(f"{i+j+1:2}. {available_classes[i+j]:<15}" for j in range(columns) if i+j < len(available_classes)))

    user_input = input("\nEnter class numbers (comma-separated) or 'ALL' [default ALL]: ").strip()
    if user_input.lower() == 'all' or user_input == '':
        return available_classes
    try:
        indices = [int(idx) - 1 for idx in user_input.split(",")]
        selected = [available_classes[idx] for idx in indices if 0 <= idx < len(available_classes)]
        if not selected:
            print("No valid classes selected. Using all classes.")
            return available_classes
        return selected
    except:
        print("Invalid input. Using all classes.")
        return available_classes

def get_color(label):
    hash_value = hash(label) % 0xFFFFFF
    return (hash_value & 0xFF, (hash_value >> 8) & 0xFF, (hash_value >> 16) & 0xFF)

def get_screen_resolution():
    try:
        monitor = get_monitors()[0]
        return monitor.width, monitor.height
    except:
        return 1280, 720

def letterbox_frame(frame, screen_w, screen_h):
    frame_h, frame_w = frame.shape[:2]
    scale = min(screen_w / frame_w, screen_h / frame_h)
    new_w, new_h = int(frame_w * scale), int(frame_h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
    return canvas

def main():
    clear_screen()
    ascii_header()

    model_file = select_model()
    model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_file}"
    download_weights(model_file, model_url)
    device = select_device()

    print(f"\n>>> Loading model '{model_file}' on device '{device}'...")
    model = YOLO(model_file)
    model.to(device)

    available_classes = list(model.names.values())
    selected_classes = select_classes(available_classes)
    print(f"\n>>> Detecting classes: {', '.join(selected_classes)}")

    try:
        conf_threshold = float(input("\nEnter confidence threshold [default 0.25]: ").strip())
        if not 0.01 <= conf_threshold <= 1.0:
            print("Out of range. Using default 0.25.")
            conf_threshold = 0.25
    except:
        conf_threshold = 0.25

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open USB camera.")
        return

    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n>>> Camera resolution: {cam_width}x{cam_height}")

    screen_width, screen_height = get_screen_resolution()
    print(f">>> Screen resolution: {screen_width}x{screen_height}")

    cv2.namedWindow('YOLOv8 Object Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('YOLOv8 Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n>>> Starting detection... Press 'q' or 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, verbose=False, conf=conf_threshold)
        detections = results[0].boxes

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            label = model.names[int(detection.cls)]
            confidence = detection.conf[0].item()
            if label not in selected_classes:
                continue
            color = get_color(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f'{label}: {confidence:.2f}'
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (x1, label_y - text_h - 5), (x1 + text_w, label_y + baseline), color, -1)
            cv2.putText(frame, label_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        display_frame = letterbox_frame(frame, screen_width, screen_height)
        cv2.imshow('YOLOv8 Object Detection', display_frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(">>> Detection ended. Goodbye!")

if __name__ == "__main__":
    main()
