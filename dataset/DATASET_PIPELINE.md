# DeepTrace Dataset Pipeline

## Overview

This directory contains the Python scripts used for dataset loading, facial landmark detection, feature extraction, machine learning dataset generation, model training, evaluation, batch benchmarking, prediction, and explainability.

The project uses the Hugging Face dataset:
`Hemg/deepfake-and-real-images`

The dataset contains two classes:
- **Fake**: Synthetically generated and altered facial images.
- **Real**: Authentic human portrait images.

---

## Dataset Processing Flow

```text
Hugging Face Dataset (Hemg/deepfake-and-real-images)
        ↓
generate_feature_dataset.py (MediaPipe 478 landmarks → 22 features)
        ↓
processed/features_test.csv (1,946 samples)
        ↓
train_model.py (Random Forest, 200 estimators, balanced)
        ↓
models/deeptrace_random_forest.joblib
        ↓
evaluate_model.py (5-Fold Stratified CV, ROC-AUC: 0.9122)
        ↓
batch_test.py (Validation on unseen samples & edge cases)
        ↓
predict_image.py (--json CLI / Express API integration)
        ↓
explain_prediction.py (Feature importance & forensic reasoning)
```

---

## Feature Vector (22 Features)

Following stability optimization (removal of `mouth_aspect_ratio` and updated outer lip landmarks `0` and `17`), the active feature set comprises:

| # | Feature Name | Description |
|---|---|---|
| 1 | `left_eye_width` | Euclidean distance between left eye corners (33, 133) |
| 2 | `right_eye_width` | Euclidean distance between right eye corners (362, 263) |
| 3 | `left_eye_height` | Vertical left eyelid distance (159, 145) |
| 4 | `right_eye_height` | Vertical right eyelid distance (386, 374) |
| 5 | `eye_distance` | Inter-pupillary / inner corner distance (33, 362) |
| 6 | `mouth_width` | Outer mouth corners distance (61, 291) |
| 7 | `mouth_height` | Outer lip distance from landmark 0 to 17 |
| 8 | `face_width` | Cheekbone boundary width (234, 454) |
| 9 | `face_height` | Forehead top to chin bottom (10, 152) |
| 10 | `nose_width` | Outer nostril distance (129, 358) |
| 11 | `nose_to_mouth` | Nose tip to upper lip border (1, 13) |
| 12 | `eyebrow_distance` | Inner eyebrow separation distance (70, 300) |
| 13 | `left_eye_aspect_ratio` | Left eye width / height ratio |
| 14 | `right_eye_aspect_ratio` | Right eye width / height ratio |
| 15 | `face_aspect_ratio` | Face width / face height ratio |
| 16 | `nose_width_ratio` | Nose width normalized by face width |
| 17 | `nose_mouth_ratio` | Nose-to-mouth distance normalized by face height |
| 18 | `eye_width_symmetry` | Bilateral eye width variance |
| 19 | `eye_height_symmetry` | Bilateral eye opening height variance |
| 20 | `eye_aspect_symmetry` | Bilateral eye aspect ratio variance |
| 21 | `eyebrow_symmetry` | Left-right eyebrow height variance relative to forehead |
| 22 | `nose_angle` | 3D angle formed by nostrils and nasal apex |

---

## Model Evaluation Results

- **5-Fold Cross-Validation Accuracy:** **83.71% (±3.19%)**
- **5-Fold Cross-Validation ROC-AUC:** **0.9122 (±0.0196)**
- **Held-out Test Precision:** **82.41%**
- **Held-out Test Recall:** **83.67%**
- **Edge-Case Non-Face Handling:** **100%** graceful detection
- **Mean Inference Latency:** **19.0 ms** per face

---

## CLI Execution Guide

From the `dataset/` directory:

```bash
# 1. Evaluate production model & generate ROC curve
py evaluate_model.py

# 2. Run batch testing on unseen samples & edge cases
py batch_test.py

# 3. Predict an individual image (formatted CLI output)
py predict_image.py test_images/real_original.jpg

# 4. Predict an individual image (JSON payload for API)
py predict_image.py --json test_images/fake_original.jpg

# 5. Run explainability breakdown
py explain_prediction.py test_images/real_original.jpg
```