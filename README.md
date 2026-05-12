# Padel Match Analyzer

An AI-powered computer vision pipeline for analyzing padel matches using object detection, player tracking, pose estimation, and rule-based analytics. The system detects players, tracks movement, identifies shots (Forehand, Backhand, Smash), and counts ball bounces from match footage.

---

## Demo Video

Final Output Video:  
https://drive.google.com/file/d/1d48FNVkZQT3r4pnsu8_BURLxg1Q0kgHl/view?usp=sharing


---

# Project Overview

This project combines deep learning models and rule-based logic to analyze padel gameplay automatically from video footage.

The pipeline performs:

- Player Detection
- Ball Detection
- Racket Detection
- Multi-Object Tracking
- Player Re-Identification
- Pose Estimation
- Shot Classification
- Bounce Detection
- Match Analytics Logging

The final output includes:
- Annotated output video
- Structured CSV and JSON analytics
- Match summary statistics

---

# Methodology

## 1. Object Detection using YOLOv8

The system uses a pre-trained YOLOv8 Nano model from Ultralytics for detecting:

- Person
- Sports Ball
- Tennis Racket

YOLO serves as the primary perception layer for locating active gameplay objects in each frame.

### Why YOLOv8?
- Fast inference speed
- Lightweight model
- Suitable for near real-time applications
- High detection accuracy for sports objects

---

## 2. Multi-Object Tracking and Re-Identification

To maintain consistent player identities across frames, the project uses:

- BotSort Tracker
- OSNet ReID Model

### Purpose
Traditional tracking methods fail during:
- Occlusions
- Rapid player movements
- Crossing trajectories

OSNet extracts visual appearance embeddings from players, allowing the tracker to re-identify players reliably even after temporary disappearance.

### Output
Each detected player receives a persistent Player ID throughout the match.

---

## 3. Pose Estimation and Shot Classification

The project integrates MediaPipe Pose Landmarker to estimate 33 body landmarks for the primary detected player.

### Shot Classification Logic

A heuristic-based approach classifies shots using relative landmark positions.

#### Smash
Detected when:
- Right wrist is above the nose landmark

#### Backhand
Detected when:
- Right wrist crosses the body centerline
- Wrist position is left of the left shoulder

#### Forehand
Detected when:
- Wrist remains on the dominant side during swing motion

### Debouncing Logic
A cooldown timer is implemented to prevent:
- Duplicate detections
- Multiple counts from a single swing motion

---

## 4. Ball Bounce Detection

Bounce detection is implemented using rule-based trajectory analysis.

### Method
The system stores a rolling history of the ball’s vertical (Y-axis) position.

A bounce is detected when:
- Ball moves downward
- Reaches a local peak/minimum
- Starts moving upward again

This pattern is analyzed over a 5-frame sliding window.

### Benefits
- Lightweight implementation
- No additional training required
- Works effectively for standard gameplay footage

---

# Challenges Faced

## 1. Motion Blur

Padel balls and rackets move extremely fast, causing:
- Blurred detections
- Missed YOLO predictions
- Unstable tracking

This especially affected ball-racket interaction analysis.

### Solution
Instead of relying purely on collision-based detection, pose-based shot classification was integrated.

---

## 2. Reliable Shot Detection

Initially, shot classification depended on:
- Ball proximity
- Racket overlap

However, inconsistent object detection reduced reliability.

### Solution
MediaPipe pose estimation was executed continuously on the full frame, improving shot consistency.

---

## 3. Perspective and Camera Angle Issues

Since the system operates in 2D space:
- Body rotation can distort landmark interpretation
- Forehand and backhand classification may occasionally flip

This limitation is caused by monocular camera depth ambiguity.

---

# Future Improvements

## 1. Audio-Based Impact Detection

Integrating audio analysis could detect:
- Racket-ball contact
- Bounce sounds

This would significantly improve shot timing precision.

---

## 2. Court Mapping and Homography

Applying homography transformation could:
- Convert gameplay into top-down court view
- Generate player heatmaps
- Analyze positioning strategies

---

## 3. Custom-Trained YOLO Model

Training a dedicated padel dataset would improve:
- Ball detection
- Racket detection
- Small object recognition under motion blur

---

## 4. Advanced Shot Classification

Future versions could classify:
- Volley
- Bandeja
- Vibora
- Lob
- Serve

using temporal sequence models and 3D pose estimation.

---

# Technologies Used

## Frameworks and Libraries

- Python
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- MediaPipe
- NumPy
- Pandas
- BoxMOT

---

# Project Structure

```bash
Padel-Match-Analyzer/
│
├── input_sample_video.mp4
├── final_output_video.mp4
├── main.py
├── requirements.txt
├── README.md
│
├── yolov8n.pt
├── pose_landmarker_lite.task
├── osnet_x1_0_msmt17.pt
│
├── shot_analysis.csv
├── shot_analysis.json
├── match_summary.json

```

---

# Setup and Installation

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- pip package manager

---

## 1. Clone Repository

```bash
git clone http://github.com/Dipin-Adhikari/cv
cd cv
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download Model Files

Download the following model file and place them inside the root folder:

- `pose_landmarker_lite.task`

---


---

# Running the Project

```bash
python main.py
```

---

# Output Files

After execution, the following files are generated:

| File Name | Description |
|---|---|
| `final_output_video.mp4` | Annotated gameplay video |
| `shot_analysis.csv` | Structured frame-by-frame analytics |
| `shot_analysis.json` | JSON version of analytics |
| `match_summary.json` | Overall match statistics |

---

# Sample Analytics Logged

## Shot Analytics

- Frame Number
- Timestamp
- Player ID
- Shot Type
- Bounce Count

## Match Summary

- Total Bounces
- Total Forehands
- Total Backhands
- Total Smashes

---

# Performance Notes

- GPU acceleration significantly improves processing speed.
- CPU execution is supported but slower.
- YOLOv8 Nano was selected for balancing:
  - Speed
  - Accuracy
  - Lightweight deployment

---

# Conclusion

This project demonstrates a practical application of computer vision and sports analytics using modern AI frameworks. By combining detection, tracking, pose estimation, and heuristic analysis, the system provides automated insights into padel gameplay and player activity.

The project can be extended further into:
- Professional sports analytics
- Automated coaching systems
- Match statistics generation
- Real-time sports broadcasting tools

---

