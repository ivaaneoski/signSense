# SignSense - Real-Time ASL Recognition

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-<0.10.15-orange?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![NumPy](https://img.shields.io/badge/Numpy->=1.24.0-blue?logo=numpy&logoColor=white)](https://numpy.org/)

SignSense is a real-time American Sign Language (ASL) recognition desktop application built in Python. Designed as a precise, developer-focused tooling interface, it uses a live webcam feed to detect, classify, and track static alphabet signs.

The visual identity is minimal, functional, and clean—information is presented securely via carefully defined overlay panels stacked over an unmodified local webcam video feed. Built without any external ML models, its recognition engine relies entirely on MediaPipe hand landmark tracking and heuristic geometric rules.

---

## 🛠️ Built With (Technology Stack)
This project is built using the following core technologies:
* **Python 3.12**: The core programming language used for the application logic. *(Note: Python 3.12 is explicitly required because newer versions of MediaPipe on Python 3.13+ drop support for the legacy `solutions` API used in this project).*
* **OpenCV (`opencv-python`)**: Used for interfacing with the webcam, capturing real-time video frames, and rendering the custom heads-up display (HUD) overlay and interface elements.
* **Google MediaPipe**: Utilizes the legacy `mediapipe.solutions.hands` API to extract 21 3D landmarks of a hand in real-time. Pinned to `<0.10.15` to ensure API and namespace compatibility.
* **NumPy**: Used for high-performance mathematical operations, matrix manipulations, and Euclidean distance calculations between hand landmarks.
* **uv (Astral)**: Recommended for lightning-fast virtual environment creation and dependency resolution.

---

## 🚀 Step-by-Step Installation Guide

Follow these steps to get the application running on your local machine:

### 1. Prerequisites
Ensure you have the following installed on your system:
* [Python 3.12](https://www.python.org/downloads/release/python-3120/) (highly recommended to avoid MediaPipe compatibility issues)
* A working webcam connected to your computer

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/SignSense.git
cd SignSense
```

### 3. Create a Virtual Environment
It is highly recommended to isolate the project dependencies. If you use `uv`, run:
```bash
uv venv --python 3.12
```
*Alternatively, using standard Python `venv`:*
```bash
python3.12 -m venv .venv
```

### 4. Activate the Environment
* **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 5. Install Dependencies
Install the required packages strictly from the `requirements.txt` to avoid API breaking changes:
```bash
pip install -r requirements.txt
```
*(Or if using `uv`: `uv pip install -r requirements.txt`)*

### 6. Run the Application
Start the ASL recognizer by passing the Python script to your environment:
```bash
python sign_language_recognition.py
```
> **Note:** The application requires camera permissions from your OS. It defaults to camera index `0`.

---

## ⌨️ Controls & Keyboard Shortcuts

While the webcam window is focused, you can use the following standard keyboard shortcuts:

| Key | Action |
|:---:|---|
| **`Q`** / **`ESC`** | Quit the application cleanly |
| **`SPACE`** | Pause / resume the webcam feed |
| **`BACKSPACE`** | Clear the letter history bar |
| **`S`** | Save a timestamped screenshot of the current frame to `./screenshots/` |

---

## ✋ Supported Signs Reference

The application supports most static ASL alphabet letters and some common gestures.

| Category | Letters |
|---|---|
| **Single finger** | I (pinky), D (index), X (hooked index) |
| **Two fingers**   | H, U, V, R, K, L |
| **Three fingers** | W |
| **Four fingers**  | B |
| **Full hand**     | C, O, E, S, A |
| **Thumb combos**  | Y, L, T, G |
| **Pinches**       | F, O, D |
| **Special**       | ILY 🤟 (thumb + index + pinky) |

> ⚠️ **Note:** Dynamic signs (like **J** and **Z**), which require fluid hand motion to signify correctly, are formally excluded from this model as it exclusively analyzes static geometrical frames.

---

## ⚠️ Known Limitations & Ambiguities

As this application relies purely on 2D geometric and hand-state heuristic conditions, consider the following limitations:
* **2D Projection Ambiguities:** Letters sharing similar topological profiles in 2D (like U, H, and R, or C and E) might exhibit slight instability if hand angles lack depth dimension visibility on standard webcams.
* **Finger Crossovers:** The **R** sign is approximated best-effort and will classify largely based on the index and middle fingers extending, which mimics the **H** and **U** states heavily.
* **Camera Angle:** For maximum accuracy, keep your hand squared and flat directed to the camera so that lateral overlapping (e.g., crossing thumbs) is visibly clear to the MediaPipe tracker.

