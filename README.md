# DeepTrace

## Explainable Deepfake Image Detection Using Facial Landmark Geometry and Symmetry Analysis

**DeepTrace** is an explainable AI system designed to detect synthetically generated (deepfake) facial portraits. Instead of relying on opaque black-box neural networks, DeepTrace extracts **478 anatomical facial landmarks** via MediaPipe, constructs an interpretable **22-dimensional biometric feature vector** (capturing bilateral symmetry, aspect ratios, and facial proportion geometry), and classifies images using a balanced **Random Forest classifier**.

The system provides human-interpretable forensic reasoning and transparent confidence scores, complete with an **Express.js API** backend, **PostgreSQL** audit logging, and a modern **React** web application.

---

## Key Metrics & Performance

- **5-Fold Cross-Validation Accuracy:** **83.71% (±3.19%)**
- **5-Fold Cross-Validation ROC-AUC:** **0.9122 (±0.0196)**
- **Held-out Test Precision:** **82.41%**
- **Held-out Test Recall:** **83.67%**
- **Inference Latency:** **~19.0 ms** per portrait
- **Non-Face / Edge-Case Handling:** **100%** graceful fallback

---

## System Architecture

```text
Input Facial Image
        ↓
MediaPipe FaceLandmarker (478 3D Coordinates)
        ↓
Biometric Geometry & Bilateral Symmetry Extraction (22 Features)
        ↓
Random Forest Classifier (200 Estimators, Class Balanced)
        ↓
Prediction (Real / Fake) + Confidence (0-100%)
        ↓
Feature Importance & Forensic Explainability Engine
        ↓
Express.js REST API  ⟷  PostgreSQL Audit Log  ⟷  React Cyber-Forensic Web UI
```

---

## 22-Dimensional Feature Space

| Feature Name | Category | Forensic Significance |
|---|---|---|
| `left_eye_width`, `right_eye_width` | Ocular Geometry | Identifies asymmetrical eye boundary distortion |
| `left_eye_height`, `right_eye_height` | Ocular Geometry | Measures eyelid aperture discrepancy |
| `eye_distance` | Inter-ocular Distance | Validates natural interpupillary separation |
| `mouth_width`, `mouth_height` | Oral Geometry | Detects mouth boundary warping (landmarks 0 & 17) |
| `face_width`, `face_height` | Craniofacial Frame | Baseline cranial boundary proportions |
| `nose_width`, `nose_to_mouth` | Nasal Geometry | Measures nasal bridge and philtrum spacing |
| `eyebrow_distance` | Supraorbital Geometry | Checks glabellar separation |
| `left_eye_aspect_ratio`, `right_eye_aspect_ratio` | Ocular Ratios | Validates eye width-to-height proportion |
| `face_aspect_ratio` | Craniofacial Ratio | Width-to-height facial proportion |
| `nose_width_ratio`, `nose_mouth_ratio` | Proportional Ratios | Scale-invariant intra-facial proportions |
| `eye_width_symmetry`, `eye_height_symmetry` | Bilateral Symmetry | Detects left/right eye dimension disparities |
| `eye_aspect_symmetry`, `eyebrow_symmetry` | Bilateral Symmetry | Flags GAN warping in eyebrow heights & eye shapes |
| `nose_angle` | 3D Angular Orientation | Validates triangular nasal apex projection |

---

## Quickstart & Execution

### 1. Python ML Pipeline

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

Run evaluation and prediction commands:
```bash
# Evaluate model across 5-fold cross validation and generate ROC curve
py dataset/evaluate_model.py

# Run batch test suite across 22 images (including edge cases)
py dataset/batch_test.py

# Predict a single image (formatted terminal output)
py dataset/predict_image.py dataset/test_images/real_original.jpg

# Generate full explainability report
py dataset/explain_prediction.py dataset/test_images/fake_original.jpg
```

### 2. Backend Server (Node.js & Express)

Navigate to the `backend/` directory:
```bash
cd backend
npm install
node server.js
```
The backend starts at `http://localhost:5000` with endpoints:
- `POST /api/analyze`: Accepts multipart file, base64 image, or preset sample.
- `GET /api/metrics`: Returns 5-fold CV and ROC-AUC metrics.
- `GET /api/samples`: Returns available test images for instant verification.
- `GET /api/history`: Returns scan history from PostgreSQL.

### 3. Frontend Web Application (React & Vite)

Navigate to the `frontend/` directory:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser for the DeepTrace Cyber-Forensic Web Interface.