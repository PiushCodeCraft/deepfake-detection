# DeepTrace

## Explainable Deepfake Image Detection Using Facial Landmark Geometry and Symmetry Analysis

DeepTrace is an explainable deepfake image detection system that analyzes
facial landmark geometry and facial symmetry to identify potentially
manipulated facial images.

The system extracts facial landmarks from an input image, calculates
geometric features such as distances, angles, and proportions, and analyzes
facial symmetry by comparing corresponding facial regions.

The extracted features are then provided to a machine learning classifier
to predict whether the input image is likely to be Real or Deepfake.

The system also provides an explainable result by highlighting suspicious
facial regions or geometric characteristics that contributed to the prediction,
along with a confidence score.

---

## Features

- Face Detection
- Facial Landmark Extraction
- Facial Geometry Analysis
- Facial Symmetry Analysis
- Feature Fusion
- Deepfake Classification
- Explainable AI
- Confidence Score
- Web-based Detection Interface

---

## Project Workflow

Input Image
     ↓
Face Detection & Landmark Extraction
     ↓
Facial Geometry Analysis
     ↓
Symmetry Analysis
     ↓
Feature Fusion
     ↓
Deepfake Classification
     ↓
Explainable Output

---

## Technologies Used

### Programming Languages
- Python
- JavaScript

### Computer Vision
- OpenCV
- MediaPipe

### Feature Analysis
- NumPy
- SciPy

### Machine Learning
- Scikit-learn

### Explainable AI
- LIME
- SHAP
- Matplotlib

### Backend
- Flask
- REST API

### Frontend
- React.js

### Database
- PostgreSQL

### Development Tools
- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

---

## Repository Structure

```text
deepfake-detection/
│
├── backend/
├── dataset/
├── docs/
├── frontend/
├── models/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt