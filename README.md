
# YOLOv8 Real-Time Object Detection using USB Camera

## About
This repository contains a Python script that demonstrates real-time object detection using the YOLOv8 model with a USB camera. YOLOv8, developed by Ultralytics, is the latest version of the YOLO series, offering high accuracy and performance for a variety of object detection tasks.

The script provides a user-friendly interface, allowing users to choose specific objects for detection from a predefined list of classes. It also automatically detects whether a GPU is available and downloads the YOLOv8 model if it is not already present.

## Features
- **Real-Time Object Detection**: Uses OpenCV to capture video feed from the USB camera and YOLOv8 to detect objects in real-time.
- **User-Selectable Classes**: Users can choose which objects to detect from an available list of 80 classes.
- **Automatic Model Download**: If the YOLOv8 model weights file is not present, the script will automatically download it from the official Ultralytics repository.
- **GPU Acceleration**: Automatically utilizes CUDA if available, otherwise falls back to CPU for inference.

## Requirements
- **Python** 3.8 or above
- **OpenCV** (`opencv-python`)
- **PyTorch** (with CUDA support for GPU acceleration)
- **Ultralytics YOLO library** (`ultralytics`)
- **CUDA Toolkit** (required for GPU acceleration)

## CUDA Installation

### Windows 11
1. **Download CUDA Toolkit**:
   - Visit the [NVIDIA CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads).
   - Select your operating system as "Windows" and version as "Windows 11".
   - Choose the installer (e.g., `exe` file) and follow the installation instructions to install CUDA and the appropriate drivers.

2. **Install cuDNN**:
   - Visit the [NVIDIA cuDNN page](https://developer.nvidia.com/cudnn) and download the cuDNN library that matches your CUDA version.
   - Extract the contents and copy the `bin`, `include`, and `lib` directories into your CUDA Toolkit installation path (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X`).

3. **Verify CUDA Installation**:
   - Open a command prompt and run:
     ```sh
     nvcc --version
     ```
   - This should display the version of CUDA installed.

4. **Install the Correct Version of PyTorch**:
   - Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).
   - Select the CUDA version that matches your installed CUDA Toolkit, and copy the installation command.
   - Example installation command:
     ```sh
     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
     ```
   - Replace `cu117` with the appropriate version based on your CUDA installation.

### Ubuntu (Latest Version)
1. **Update System Packages**:
   - Open a terminal and run:
     ```sh
     sudo apt update
     sudo apt upgrade
     ```

2. **Install CUDA Toolkit**:
   - Visit the [NVIDIA CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads).
   - Select your operating system as "Linux" and version as "Ubuntu".
   - Follow the instructions provided for adding the NVIDIA package repository and installing CUDA.

   Example commands for installing CUDA on Ubuntu:
   ```sh
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -sr | tr -d '.')/x86_64/cuda-ubuntu$(lsb_release -sr | tr -d '.').pin
   sudo mv cuda-ubuntu$(lsb_release -sr | tr -d '.').pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -sr | tr -d '.')/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -sr | tr -d '.')/x86_64/ /"
   sudo apt update
   sudo apt install -y cuda
   ```

3. **Install cuDNN**:
   - Visit the [NVIDIA cuDNN page](https://developer.nvidia.com/cudnn) and download the cuDNN library that matches your CUDA version.
   - Extract the downloaded file and copy the `lib`, `include`, and `bin` directories to your `/usr/local/cuda/` directory.

4. **Set Up Environment Variables**:
   - Add the following lines to your `.bashrc` file:
     ```sh
     export PATH=/usr/local/cuda/bin:$PATH
     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
     ```
   - Save the file and update your terminal:
     ```sh
     source ~/.bashrc
     ```

5. **Verify CUDA Installation**:
   - Run:
     ```sh
     nvcc --version
     ```
   - This should display the version of CUDA installed.

6. **Install the Correct Version of PyTorch**:
   - Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).
   - Select the CUDA version that matches your installed CUDA Toolkit, and copy the installation command.
   - Example installation command:
     ```sh
     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
     ```
   - Replace `cu117` with the appropriate version based on your CUDA installation.

## Installation

### Step 1: Clone the Repository
Clone this repository to your local machine:
```sh
git clone https://github.com/KD5VMF/Yolov8.git
cd Yolov8
```

### Step 2: Create a Virtual Environment (Recommended)
Creating a virtual environment helps keep dependencies organized:
```sh
python -m venv yolo_env
```

Activate the virtual environment:

- On Windows:
    ```sh
    yolo_env\Scripts\activate
    ```
- On macOS/Linux:
    ```sh
    source yolo_env/bin/activate
    ```

### Step 3: Install Dependencies
Install the required dependencies using `pip`:
```sh
pip install -r requirements.txt
```

If the `requirements.txt` is not available, you can manually install the dependencies:
```sh
pip install opencv-python torch ultralytics screeninfo
```

## Usage
To run the object detection script:

### Step 1: Run the Script
```sh
python detect.py
```

### Step 2: Follow On-Screen Instructions
- The script will start by displaying a welcome message and loading the YOLOv8 model.
- You will be prompted to choose which objects to detect from a list of available classes.
- The available classes are displayed in a multi-column format for easy readability.
- Enter the numbers of the classes you want to detect, separated by commas, or type `ALL` to detect all classes.

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

### Output
- The USB camera feed will be displayed with bounding boxes drawn around the detected objects.
- The labels and confidence scores for each detected object will also be displayed.

- Press the `q` key to exit the real-time object detection.

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
