#!/bin/bash

PROJECT_DIR="$HOME/YOLOv8-GPU-Webcam"
VENV_DIR="$PROJECT_DIR/envGPU"
APP_PY="$PROJECT_DIR/YOLO-GPU-Webcam-App.py"
HTML_DIR="$PROJECT_DIR/templates"
HTML_FILE="$HTML_DIR/index.html"

echo "📁 Creating project folder at $PROJECT_DIR..."
mkdir -p "$HTML_DIR"

echo "🐍 Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "📦 Installing Python dependencies in venv..."
pip install --upgrade pip
pip install flask opencv-python ultralytics

echo "📄 Creating Python application..."
cat > "$APP_PY" << 'EOF'
from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
import threading

app = Flask(__name__)
model = YOLO("yolov8n.pt")
device = "cpu"
class_names = model.names
lock = threading.Lock()
video_capture = None

def gen_frames():
    global video_capture, model, device
    try:
        video_capture = cv2.VideoCapture(0)
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            with lock:
                results = model.predict(source=frame, device=device, stream=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"{class_names[cls]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        if video_capture:
            video_capture.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, device, class_names
    if request.method == 'POST':
        model_name = request.form.get('model')
        device = request.form.get('device')
        try:
            model = YOLO(model_name)
            model.to(device)
            class_names = model.names
            print(f"🔁 Reloading model: {model_name} on {device}")
        except Exception as e:
            print(f"❌ Failed to reload: {e}")
    return render_template("index.html", model=request.form.get('model', 'yolov8n.pt'),
                           device=device, class_names=class_names)

@app.route('/video_feed')
def video_feed():
    try:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"⚠️ Stream error: {e}")
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0')
EOF

echo "🖼️ Creating HTML template..."
cat > "$HTML_FILE" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv8 Webcam App</title>
</head>
<body>
    <h1>YOLOv8 Webcam App</h1>
    <form method="POST">
        <label for="model">Model name:</label>
        <input type="text" name="model" value="yolov8n.pt">
        <label for="device">Device:</label>
        <select name="device">
            <option value="cpu">CPU</option>
            <option value="cuda">CUDA</option>
        </select>
        <button type="submit">Apply</button>
    </form>
    <br>
    <img src="{{ url_for('video_feed') }}" width="720">
</body>
</html>
EOF

echo "✅ All files created."

echo -e "\n🚀 To run the app:"
echo "  cd $PROJECT_DIR"
echo "  source envGPU/bin/activate"
echo "  python3 YOLO-GPU-Webcam-App.py"
