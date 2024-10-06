
# YOLOv8 Real-Time Object Detection using USB Camera (Web Version)

## About
This project provides a real-time object detection solution using YOLOv8 with a USB camera, specifically adapted for headless Ubuntu servers. The script processes the video feed and serves it via a local web server using Flask, allowing users to view detected objects on a web page.

The solution leverages GPU acceleration if available, otherwise it falls back to the CPU.

## Features
- **Real-Time Object Detection**: Uses OpenCV to capture video from a USB camera and YOLOv8 for object detection.
- **Web-Based Output**: Instead of showing the camera feed in an OpenCV window, the detection is streamed on a webpage, making it suitable for headless environments.
- **Automatic Model Download**: Downloads YOLOv8 model weights if not available.
- **GPU Acceleration**: Uses CUDA for faster performance if a GPU is available.

## Requirements
- **Python** 3.8 or above
- **OpenCV** (`opencv-python`)
- **PyTorch** (with CUDA support for GPU acceleration)
- **Ultralytics YOLO library** (`ultralytics`)
- **Flask** for web server
- **CUDA Toolkit** (optional, for GPU acceleration)

## Installation

### Step 1: Clone the Repository
Clone this repository to your local machine:
```sh
git clone https://github.com/YourRepository/Yolov8-Web.git
cd Yolov8-Web
```

### Step 2: Create a Virtual Environment
Creating a virtual environment helps manage dependencies:
```sh
python3 -m venv envDetect
source envDetect/bin/activate
```

### Step 3: Install Dependencies
Install the required dependencies using `pip`:
```sh
pip install opencv-python torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
pip install ultralytics flask screeninfo
```

### Step 4: Set Up Permissions
Add your user to the `video` group to access USB camera devices:
```sh
sudo usermod -aG video $USER
sudo reboot
```

### Step 5: Run the Object Detection Script
Activate your virtual environment and run the script:
```sh
source envDetect/bin/activate
python detect_web.py
```

### Step 6: View the Detection Stream
Open your web browser and navigate to:
```
http://localhost:5000
```

You will see the real-time video feed with detected objects.

## Troubleshooting

1. **Camera Not Accessible**:
   - Ensure the USB camera is properly connected and recognized (`ls /dev/video*`).
   - Verify permissions by adding your user to the `video` group.

2. **CUDA Not Available**:
   - If CUDA is not detected, ensure the correct version of the NVIDIA driver is installed and `nvidia-smi` shows the GPU details.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** for providing the YOLOv8 model.
- **OpenCV** for real-time image processing.
- **PyTorch** for enabling deep learning model inference.

## Author
ChatGPT 4o

Feel free to reach out for any questions or contributions to improve this project further.
