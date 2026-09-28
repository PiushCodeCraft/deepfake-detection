# DeepTrace

## Explainable Deepfake Image Detection Using Facial Landmark Geometry and Symmetry Analysis

DeepTrace is an explainable deepfake image detection system that analyzes
facial landmark geometry and facial symmetry to identify potentially
manipulated facial images.

The system extracts facial landmarks from an input image using MediaPipe,
calculates geometric features such as distances, angles, aspect ratios,
relative proportions, and symmetry measurements, and provides these features
to a machine learning classifier.

The current machine learning implementation uses a Random Forest classifier
to predict whether an input facial image is likely to be **Real** or **Fake**.

The system also provides explainable information by showing the important
facial geometry and symmetry features associated with the prediction,
together with a confidence score.

---

## Project Objectives

- Detect potentially manipulated facial images.
- Extract facial landmarks using MediaPipe.
- Analyze facial geometry using measurable features.
- Analyze left-right facial symmetry.
- Combine geometry and symmetry features for classification.
- Provide an interpretable prediction instead of relying only on raw image
  classification.
- Provide prediction confidence and feature-based explanation.
- Integrate the detection system with a web application.
- Maintain prediction information using PostgreSQL.
- Use CI/CD practices to automatically check the project code.

---

## Features

- Face Detection
- 478-Point Facial Landmark Extraction
- Facial Geometry Analysis
- Facial Symmetry Analysis
- Feature Engineering
- Random Forest Classification
- Prediction Confidence
- Feature Importance Based Explainability
- Web-based Detection Interface
- Node.js/Express Backend
- PostgreSQL Database
- Git/GitHub Version Control
- GitHub Actions CI

---

## Project Workflow

```text
Input Image
     ↓
Face Detection
     ↓
Facial Landmark Extraction
     ↓
Facial Geometry Analysis
     ↓
Symmetry Analysis
     ↓
Feature Extraction
     ↓
Machine Learning Classification
     ↓
Prediction + Confidence
     ↓
Explainable Output
     ↓
Web Application / Database