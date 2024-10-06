
# YOLOv8 Real-Time Object Detection using USB Camera (Web Version)

## About
This repository contains a Python script that demonstrates real-time object detection using the YOLOv8 model with a USB camera. Instead of displaying the output using OpenCV windows, it hosts a web server to stream the detection results live, making it suitable for use on headless servers. The web server is built using Flask, allowing easy access from any device on the network.

Users can select which object classes to detect, and there is a mechanism to stop the video stream either by the user or by detecting if the viewer has left. This version also displays information about the YOLO model, device in use (CPU or CUDA), and other useful information as an overlay in the video stream.

## Features
- **Real-Time Object Detection**: Uses OpenCV to capture video feed from a USB camera and YOLOv8 to detect objects in real-time.
- **User-Selectable Classes**: Users can choose which objects to detect from an available list of 80 classes.
- **Automatic Model Download**: If the YOLOv8 model weights file is not present, the script will automatically download it from the official Ultralytics repository.
- **GPU Acceleration**: Automatically utilizes CUDA if available, otherwise falls back to CPU for inference.
- **Web-Based Video Stream**: Displays the output using a local web server, making it accessible from any device on the network.

## Requirements
- **Python** 3.8 or above
- **OpenCV** (`opencv-python`)
- **PyTorch** (with CUDA support for GPU acceleration)
- **Ultralytics YOLO library** (`ultralytics`)
- **Flask** (for web streaming)
- **CUDA Toolkit** (required for GPU acceleration, optional)

## Installation

### Step 1: Clone the Repository
Clone this repository to your local machine:
```sh
git clone https://github.com/KD5VMF/Yolov8-Web.git
cd Yolov8-Web
```

### Step 2: Create a Virtual Environment (Recommended)
Creating a virtual environment helps keep dependencies organized:
```sh
python3 -m venv envDetect
```

Activate the virtual environment:

- On Linux/macOS:
    ```sh
    source envDetect/bin/activate
    ```
- On Windows:
    ```sh
    envDetect\Scripts\activate
    ```

### Step 3: Install Dependencies
Install the required dependencies using `pip`:
```sh
pip install -r requirements.txt
```

If the `requirements.txt` is not available, you can manually install the dependencies:
```sh
pip install opencv-python torch torchvision torchaudio ultralytics flask screeninfo
```

### Step 4: Verify CUDA Installation (Optional for GPU Acceleration)
- Ensure that you have installed CUDA correctly by running:
  ```sh
  nvcc --version
  ```
- Install the correct version of PyTorch that matches your CUDA version. You can find the appropriate installation command [here](https://pytorch.org/get-started/locally/).

## Usage

### Step 1: Run the Script
Run the object detection script using the following command:
```sh
python detect_web.py
```

### Step 2: Access the Web Interface
- Once the script is running, you will see the following output:
  ```
  Starting Flask server... Access the video stream at http://localhost:5000
  ```
- Open your browser and navigate to `http://<server-ip>:5000` to access the object detection interface.

### Step 3: Select Classes for Detection
- The web page will display a list of available classes for detection in an organized manner.
- Enter the numbers of the classes you want to detect, separated by commas, or type `ALL` to detect all classes.

### Step 4: View Real-Time Detection
- After selecting the classes, the real-time video feed will start, showing detected objects with bounding boxes and labels.
- The video will have overlays at the top and bottom displaying information about the model and the device in use (CPU or CUDA).

### Example Usage
1. **Select Specific Classes for Detection**:
    ```
    Enter the numbers of the classes you want to detect (comma-separated) or type 'ALL' for all classes: 1, 3, 7
    ```
   This will enable detection of "person," "car," and "train."

2. **Detect All Classes**:
    ```
    Enter the numbers of the classes you want to detect (comma-separated) or type 'ALL' for all classes: ALL
    ```
   This will enable detection of all available objects.

### Step 5: Stop the Stream
- You can stop the server by navigating to the `/stop` endpoint:
  ```
  http://<server-ip>:5000/stop
  ```

## Troubleshooting
1. **Camera Not Opening**:
   - Ensure that your USB camera is connected properly.
   - Verify that the camera is not being used by another application.

2. **Model Download Issues**:
   - If the model weights file fails to download, manually download `yolov8x.pt` from the [Ultralytics GitHub releases page](https://github.com/ultralytics/assets/releases) and place it in the same directory as the script.

3. **CUDA Not Available**:
   - If CUDA is not detected, the script will default to CPU for inference. Ensure you have installed the correct version of PyTorch with GPU support as described above.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** for providing the YOLOv8 model.
- **OpenCV** for real-time image processing.
- **PyTorch** for enabling deep learning model inference.

## Author
ChatGTP 4o

Feel free to reach out for any questions or contributions to improve this project further.
