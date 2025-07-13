#!/bin/bash

APP_DIR=~/YOLOv8-GPU-Webcam
VENV_DIR=~/envGPU
APP_FILE="$APP_DIR/YOLO-GPU-Webcam-App.py"
TEMPLATE_DIR="$APP_DIR/templates"
TEMPLATE_FILE="$TEMPLATE_DIR/index.html"

echo "📁 Creating project at $APP_DIR..."
mkdir -p "$TEMPLATE_DIR"

echo "🐍 Creating Python virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install flask opencv-python ultralytics

echo "📄 Writing main Python app..."
cat > "$APP_FILE" << 'EOF'
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import threading

app = Flask(__name__)

model_name = "yolov8n.pt"
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
model = YOLO(model_name).to(device)
video_capture = None
lock = threading.Lock()

def gen_frames():
    global video_capture
    try:
        with lock:
            if video_capture is not None:
                video_capture.release()
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                raise RuntimeError("Cannot access webcam.")

        while True:
            with lock:
                success, frame = video_capture.read()
            if not success:
                break

            results = model(frame)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print(f"⚠️  Stream error: {e}")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"🔁 Starting {model_name} on {device}")
    app.run(host='0.0.0.0', port=5000)
EOF

echo "🖼️ Writing HTML template..."
cat > "$TEMPLATE_FILE" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>YOLOv8 Stream</title>
    <style>
        body {
            margin: 0;
            background: #000;
            overflow: hidden;
        }
        img {
            display: block;
            width: 100vw;
            height: 100vh;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <img src="{{ url_for('video_feed') }}" alt="Webcam Stream">
</body>
</html>
EOF

echo "✅ Setup complete!"
echo "➡️  To start the app:"
echo "   source $VENV_DIR/bin/activate"
echo "   python3 $APP_FILE"
