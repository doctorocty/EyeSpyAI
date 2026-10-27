# EyeSpyAI

<p align="center">
  <strong>Real-Time Computer Vision for Object and Facial Expression Detection</strong>
</p>

<p align="center">
  A Python-based computer vision project exploring real-time visual recognition through webcam-based object detection and facial expression analysis.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-536878?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLOv8-334E68?style=flat-square" alt="YOLOv8">
  <img src="https://img.shields.io/badge/FER-526D82?style=flat-square" alt="FER">
  <img src="https://img.shields.io/badge/License-MIT-64748B?style=flat-square" alt="MIT License">
</p>

<p align="center">
  <a href="https://github.com/doctorocty/EyeSpyAI">Source Code</a>
  &nbsp; | &nbsp;
  <a href="https://github.com/doctorocty/EyeSpyAI/issues">Report an Issue</a>
  &nbsp; | &nbsp;
  <a href="LICENSE">MIT License</a>
</p>

---

## Overview

EyeSpyAI is a Python-based computer vision project that brings together two independent machine learning applications: real-time object detection and facial expression recognition.

Using a webcam as the video input, the project processes incoming frames, identifies supported objects or visible faces, and overlays the model's predictions directly onto the video feed.

EyeSpyAI demonstrates practical applications of computer vision, model inference, and real-time image processing using established machine learning libraries.

## Features

### 1. Real-Time Object Detection

The object detection module uses Ultralytics YOLOv8 Nano to identify objects in a live webcam feed.

- Detects objects supported by the pretrained model.
- Draws bounding boxes around detected objects.
- Displays predicted class labels.
- Shows confidence scores alongside detection labels.
- Continuously processes video frames.
- Supports exiting the detection window using the `Q` key.

**Implementation:** `object_detect.py`

### 2. Facial Expression Recognition

The facial expression module uses the FER library to identify faces and estimate their associated facial expressions.

- Detects faces within webcam frames.
- Predicts facial expression categories.
- Identifies the highest-scoring expression for each detected face.
- Displays face bounding boxes and expression labels.
- Processes video frames continuously.
- Provides console feedback during initialization and shutdown.

**Implementation:** `emotion_detect.py`

The predictions describe visible facial-expression patterns. They should not be interpreted as definitive evidence of a person's actual emotional state.

### 3. Independent Processing Pipelines

Each detection feature operates through its own Python script. This keeps the implementation simple and allows either module to run independently.

Object detection and facial expression recognition can therefore be tested separately without requiring a combined application interface.

### 4. Local Webcam Input

Both modules capture video through a connected webcam and process the frames locally.

The current implementation is intended for real-time webcam demonstrations. Processing uploaded images or prerecorded videos would require additional code.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam access, frame capture, image processing, and annotations |
| Ultralytics YOLOv8 | Object detection and classification |
| FER | Facial expression recognition |
| TensorFlow | Machine learning backend for emotion recognition |
| MediaPipe | Supporting computer vision functionality used by the dependency stack |
| NumPy | Numerical operations and array processing |
| Protobuf | Supporting dependency for machine learning libraries |
| MoviePy | Dependency used by the FER package |

## How It Works

EyeSpyAI contains two distinct processing pipelines.

### Object Detection Workflow

1. Initialize the YOLOv8 Nano model.
2. Open the webcam and capture a frame.
3. Pass the frame to the object detection model.
4. Retrieve the predicted object classes, bounding boxes, and confidence scores.
5. Draw bounding boxes and labels on the frame.
6. Display the processed frame in a window.
7. Continue until the camera feed stops or the user presses `Q`.

### Facial Expression Recognition Workflow

1. Initialize the FER detector with MTCNN-based face detection enabled.
2. Open the webcam and capture a frame.
3. Detect faces and estimate their expression scores.
4. Select the highest-scoring expression for each detected face.
5. Draw a bounding box and expression label.
6. Display the processed frame.
7. Continue until the camera feed stops or the user presses `Q`.

## Getting Started

Follow the steps below to run EyeSpyAI locally.

### Prerequisites

Make sure you have:

- Python 3.10 or 3.11.
- Git.
- pip.
- A working webcam.
- A compatible operating system and machine learning environment.

Python 3.10 or 3.11 is recommended for the dependency combination described below. Package compatibility can vary depending on the Python version and operating system.

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/doctorocty/EyeSpyAI.git
```

Navigate into the project directory:

```bash
cd EyeSpyAI
```

### 2. Create a Virtual Environment

A virtual environment keeps the project's dependencies separate from other Python installations.

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

The current `requirements.txt` contains explanatory dependency notes instead of standard pip requirement entries. Until that file is updated, install the packages directly:

```bash
python -m pip install --upgrade pip
```

```bash
pip install opencv-contrib-python ultralytics fer moviepy "tensorflow==2.15.1" "mediapipe==0.10.21" "numpy<2" "protobuf==4.25.8"
```

These versions are a proposed starting point for a compatible environment, not a verified lockfile. Dependency resolution may require adjustments for your operating system and Python installation.

For reproducible installations, maintain a valid `requirements.txt` containing tested package versions.

### 4. Run Object Detection

Execute the following command from the project directory:

```bash
python object_detect.py
```

The webcam window should open and display detected objects with their labels and confidence scores.

Press `Q` while the window is active to exit.

The script loads `yolov8n.pt`. Keep the model weights in the project directory, or allow Ultralytics to download the named pretrained weights when supported and an internet connection is available.

### 5. Run Facial Expression Recognition

Close the object detection application, then execute:

```bash
python emotion_detect.py
```

The script initializes the FER detector and attempts to open the webcam. Detected faces will be highlighted with their predicted expression labels.

Press `Q` to exit.

**Note:** Run the scripts separately. The current implementation does not provide a single launcher that starts both detection pipelines.

## Project Structure

```text
EyeSpyAI/
├── object_detect.py    # YOLOv8 object detection
├── emotion_detect.py   # Facial expression recognition
├── yolov8n.pt          # YOLOv8 Nano model weights
├── requirements.txt    # Dependency documentation
├── .gitignore          # Git ignore rules
├── LICENSE             # MIT License
└── README.md           # Project documentation
```

## Troubleshooting

### Webcam Not Opening

Check that your webcam is connected and accessible, and that other applications are not using it.

The object detection script uses OpenCV's default camera backend. The facial expression script explicitly uses the Microsoft Media Foundation backend, which is intended for Windows.

On macOS or Linux, replace:

```python
cv2.VideoCapture(0, cv2.CAP_MSMF)
```

with:

```python
cv2.VideoCapture(0)
```

if the existing backend is unavailable.

### Missing Model Weights

Ensure that `yolov8n.pt` is accessible from the project's working directory. If the weights are not present, the Ultralytics library may attempt to download them automatically when the model is initialized.

### Dependency Conflicts

TensorFlow, MediaPipe, NumPy, and Protobuf have interdependent version requirements.

If installation fails:

- Confirm your Python version.
- Use a fresh virtual environment.
- Review the package versions reported in the error.
- Avoid installing multiple OpenCV distributions into the same environment unnecessarily.

### Slow Inference

Processing speed depends on hardware, model initialization, frame resolution, and the installed machine learning libraries.

Closing resource-intensive applications or using a machine with better processing capabilities may improve performance.

## Limitations and Responsible Use

- The current detection scripts operate on live webcam feeds.
- Object recognition is limited to classes supported by the pretrained YOLOv8 Nano model.
- Facial expression predictions can be affected by lighting, camera quality, facial pose, and other visual conditions.
- Facial expression recognition cannot reliably establish what someone is actually feeling or thinking.
- Webcam analysis should be conducted with appropriate consent and respect for personal privacy.
- Third-party packages and pretrained model weights may have licensing requirements separate from this repository's license.

EyeSpyAI is intended as an educational and experimental computer vision project. Its outputs should not be treated as infallible or used as the sole basis for consequential decisions.

## Future Improvements

Potential directions for future development include:

- Support for static images and prerecorded video files.
- A unified interface for running both detection modules.
- Configurable confidence thresholds.
- Performance measurements, including inference time and frames per second.
- Improved error handling for camera initialization and model loading.
- A tested, version-pinned dependency file for easier setup.
- Additional options for controlling video capture and processing.

These are potential enhancements rather than features currently guaranteed by the implementation.

## Contributing

Contributions, suggestions, and bug reports are welcome.

To contribute:

1. Fork the repository.
2. Create a branch for your changes.
3. Implement and test your improvements.
4. Commit your changes with a descriptive message.
5. Open a pull request explaining your contribution.

For significant changes, consider opening an issue first to discuss the proposed approach.

## License

EyeSpyAI is licensed under the **MIT License**.

The MIT License permits use, copying, modification, distribution, sublicensing, and selling copies of the software, subject to its conditions.

Redistributions of the software or substantial portions of it must include the applicable copyright notice and permission notice.

**Copyright (c) 2026 Sarah Fatima Lari**

See the [`LICENSE`](LICENSE) file for the complete license terms.

The MIT License for this repository does not override the separate terms that may apply to third-party libraries or pretrained model weights.

## Author

**Sarah Fatima Lari**

- GitHub: [@doctorocty](https://github.com/doctorocty)
- Portfolio: [sarahs-portfolio-inky.vercel.app](https://sarahs-portfolio-inky.vercel.app/)

---

<p align="center">
  <sub>Exploring computer vision through real-time detection and machine learning.</sub>
</p>
