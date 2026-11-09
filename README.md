# AirCanvas Studio
### Real-Time Virtual Drawing System Using Deep Learning-Based Hand Gesture Recognition

AirCanvas Studio is a real-time virtual drawing application that allows users to draw, erase and create digital artwork using hand gestures in front of a webcam, without requiring a physical mouse, touchscreen or drawing tablet.

The system uses MediaPipe Hands for real-time hand landmark detection and gesture interpretation. It also includes a custom 1D Convolutional Neural Network (CNN) trained on hand landmark data for gesture classification.

The application provides an interactive drawing workspace with multiple brush styles, colour selection, shape tools and gesture-based controls.

---

## Features

### 1. Real-Time Hand Tracking
- Detects and tracks hand movements using a webcam.
- Extracts 21 hand landmarks using MediaPipe Hands.
- Tracks the index fingertip to control the drawing cursor.
- Uses hand landmark positions and finger configurations to identify drawing actions.

### 2. Gesture-Based Drawing
- **Index finger raised:** Enables drawing.
- **Index finger bent:** Stops drawing.
- **Other fingers raised:** Stops drawing.
- **Open palm:** Automatically activates the large-area eraser.
- **Pause:** Temporarily disables drawing interactions.

### 3. Drawing Workspace
- Screen Drawing Mode: Draw directly over the live webcam feed.
- Canvas Mode: View the live camera and a separate digital drawing canvas.
- Supports freehand drawing and geometric shapes.

### 4. Creative Brush Collection

AirCanvas includes multiple brush presets:

- Classic Pen
- Fine Liner
- Pencil
- Soft Sketch
- Marker
- Highlighter
- Calligraphy
- Spray

Users can customise brush size and opacity.

### 5. Colour Studio
- Wide range of selectable colours.
- Multiple shades across different colour families.
- Grayscale colour selection.
- Custom colour picker.
- Quick-access colour palette.

### 6. Drawing Management
- Undo and Redo.
- Clear the drawing canvas.
- Save artwork as an image.
- Large-area palm erasing for removing unwanted strokes.

---

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe Hands |
| Deep Learning | TensorFlow, Keras |
| Neural Network | 1D Convolutional Neural Network |
| Numerical Processing | NumPy |
| GUI Development | Tkinter |
| Image Processing | Pillow |
| Data Visualisation | Matplotlib |
| Machine Learning Evaluation | Scikit-learn |

---

## System Architecture

The AirCanvas system follows a sequence of hand detection, landmark extraction, gesture interpretation and drawing operations.

```text
             WEBCAM INPUT
                  |
                  v
          IMAGE PROCESSING
             (OpenCV)
                  |
                  v
        HAND LANDMARK DETECTION
           (MediaPipe Hands)
                  |
                  v
       21 HAND LANDMARK EXTRACTION
                  |
                  v
          GESTURE INTERPRETATION
                  |
          +-------+--------+
          |                |
          v                v
    INDEX FINGER       OPEN PALM
       DETECTED         DETECTED
          |                |
          v                v
    DRAWING MODE       AUTO ERASER
          |                |
          +-------+--------+
                  |
                  v
          DRAWING WORKSPACE
                  |
                  v
        DISPLAY / SAVE ARTWORK
```

### Deep Learning Model

A custom 1D CNN model is included for hand gesture classification.

The model processes normalised hand landmark coordinates representing 21 hand landmarks, each containing three coordinates (X, Y and Z).

**CNN Architecture:**

1. Input Layer: 21 × 3 landmark features.
2. Conv1D Layer: 32 filters with ReLU activation.
3. MaxPooling1D Layer.
4. Conv1D Layer: 64 filters with ReLU activation.
5. Global Average Pooling Layer.
6. Dense Layer: 64 neurons with ReLU activation.
7. Dropout Layer: 0.3.
8. Output Layer: 3 classes with Softmax activation.

The model is trained to classify the following gestures:

- Draw
- Stop
- Erase

The application also uses MediaPipe landmark geometry for responsive fingertip tracking and direct drawing controls.

---

## Project Structure

```text
AirCanvas-Studio/
│
├── app.py
│
├── collect_data.py
├── train_gesture.py
├── test_gesture.py
│
├── drawing.py
├── digit_recognition.py
│
├── requirements.txt
├── .gitignore
├── README.md
│
├── dataset/
│   └── gestures/
│       ├── draw/
│       ├── stop/
│       └── erase/
│
├── models/
│   ├── gesture_model.keras
│   └── digit_model.keras
│
└── drawings/
```

---

## Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Webcam
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/afrinsulthana2/AirCanvas-Studio.git
```

### Step 2: Navigate to the Project Directory

```bash
cd AirCanvas-Studio
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
venv\Scripts\Activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python app.py
```

Allow webcam access when prompted.

---

## Gesture Controls

| Hand Gesture | Action |
|---|---|
| Index finger straight, other three folded | Draw |
| Index finger bent | Stop drawing |
| Any other finger raised | Stop drawing |
| Fully open palm | Automatically erase |
| Pause button | Pause interactions |

**Note:** Palm erasing works automatically when an open palm is detected. The Eraser toolbar option is also available for manual tool selection.

---

## Dataset and Model Training

The project includes a custom hand gesture dataset collected using MediaPipe landmark extraction.

### Dataset Collection

Run:

```bash
python collect_data.py
```

The collection script captures hand landmark information and saves the samples as NumPy files.

### Model Training

Run:

```bash
python train_gesture.py
```

The training script:
- Loads the collected gesture dataset.
- Prepares and splits the data.
- Trains the 1D CNN model.
- Evaluates its classification performance.
- Saves the trained model in the `models/` directory.

### Gesture Testing

Run:

```bash
python test_gesture.py
```

This script provides a way to test gesture recognition using the trained model and webcam input.

---

## Project Objectives

- Develop a touch-free virtual drawing environment.
- Implement real-time hand tracking and landmark detection.
- Apply deep learning techniques for hand gesture classification.
- Enable natural drawing interactions using finger movements.
- Provide an interactive workspace with creative drawing tools.
- Explore human-computer interaction through computer vision.

---

## Future Enhancements

- Improve gesture recognition accuracy under different lighting conditions.
- Add support for multiple-hand tracking.
- Introduce additional drawing gestures.
- Integrate AI-powered sketch recognition.
- Implement handwriting and digit recognition.
- Add cloud-based artwork storage.
- Develop a web-based version of AirCanvas.

---

## Author

**AirCanvas Studio**

Developed as a Deep Learning mini-project.

**GitHub Repository:**  
https://github.com/afrinsulthana2/AirCanvas-Studio

---

If you find this project interesting, consider giving the repository a star.