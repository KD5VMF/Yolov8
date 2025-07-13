#!/bin/bash

PROJECT_DIR=~/YOLOv8-GPU-Webcam
VENV_DIR=~/envGPU

echo "📁 Creating project folder..."
mkdir -p "$PROJECT_DIR/templates"
cd "$PROJECT_DIR" || exit 1

echo "🐍 Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "📦 Installing required packages..."
pip install --upgrade pip
pip install flask opencv-python ultralytics

echo "📄 Creating Python app..."
cat > YOLO-GPU-Webcam-App.py << 'EOF'
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

model_name = "yolov8x.pt"
device = "cuda"
model = YOLO(model_name).to(device)
cap = cv2.VideoCapture(0)

def generate_frames():
    prev_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        info = f"{model_name} on {device.upper()} | FPS: {fps:.2f}"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF

echo "🖼️ Creating responsive HTML template..."
cat > templates/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv8 Webcam Viewer</title>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            background-color: black;
            color: white;
            font-family: sans-serif;
            height: 100%;
            overflow: hidden;
        }
        body {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        h1 {
            margin: 10px 0;
            font-size: 1.2em;
        }
        #video-wrapper {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 100%;
        }
        #video-stream {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <h1>📸 YOLOv8X Object Detection (Auto Fit)</h1>
    <div id="video-wrapper">
        <img id="video-stream" src="{{ url_for('video_feed') }}">
    </div>
</body>
</html>
EOF

echo "✅ Setup complete!"
echo "👉 To run:"
echo "    source $VENV_DIR/bin/activate"
echo "    cd $PROJECT_DIR"
echo "    python3 YOLO-GPU-Webcam-App.py"
