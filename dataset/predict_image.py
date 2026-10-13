import os
import sys

# Suppress MediaPipe/TF C++ log output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib


import json
from pathlib import Path

# ==============================
# SETTINGS & PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "deeptrace_random_forest.joblib"
DEFAULT_LANDMARK_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"

MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
LANDMARK_MODEL = Path(os.environ.get("LANDMARK_MODEL", DEFAULT_LANDMARK_MODEL))

# Check for --json flag
JSON_OUTPUT = "--json" in sys.argv
args = [arg for arg in sys.argv[1:] if arg != "--json"]

if len(args) < 1:
    if JSON_OUTPUT:
        print(json.dumps({"status": "error", "message": "Missing image_path argument"}))
    else:
        print("Usage: py predict_image.py [--json] <image_path>")
    sys.exit(1)

IMAGE_PATH = args[0]

if not os.path.exists(IMAGE_PATH):
    if JSON_OUTPUT:
        print(json.dumps({"status": "error", "message": f"Image not found: {IMAGE_PATH}"}))
    else:
        print(f"Error: Image not found: {IMAGE_PATH}")
    sys.exit(1)

if not JSON_OUTPUT:
    print("Loading DeepTrace model...")

model = joblib.load(str(MODEL_PATH))

if not JSON_OUTPUT:
    print("Model loaded successfully.")


# ==============================
# MEDIAPIPE SETUP
# ==============================

base_options = python.BaseOptions(
    model_asset_path=str(LANDMARK_MODEL)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(
    options
)


# ==============================
# HELPER FUNCTIONS
# ==============================

def distance(a, b):

    return np.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def angle(a, b, c):

    ba = np.array([
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ])

    bc = np.array([
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    ])

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return 0.0

    cosine = np.dot(ba, bc) / denominator

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def symmetry(a, b):

    denominator = max(a, b)

    if denominator == 0:
        return 0.0

    return abs(a - b) / denominator


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(landmarks):

    left_eye_width = distance(
        landmarks[33],
        landmarks[133]
    )

    right_eye_width = distance(
        landmarks[362],
        landmarks[263]
    )

    left_eye_height = distance(
        landmarks[159],
        landmarks[145]
    )

    right_eye_height = distance(
        landmarks[386],
        landmarks[374]
    )

    eye_distance = distance(
        landmarks[33],
        landmarks[362]
    )

    mouth_width = distance(
        landmarks[61],
        landmarks[291]
    )

    mouth_height = distance(
        landmarks[0],
        landmarks[17]
    )

    face_width = distance(
        landmarks[234],
        landmarks[454]
    )

    face_height = distance(
        landmarks[10],
        landmarks[152]
    )

    nose_width = distance(
        landmarks[129],
        landmarks[358]
    )

    nose_to_mouth = distance(
        landmarks[1],
        landmarks[13]
    )

    eyebrow_distance = distance(
        landmarks[70],
        landmarks[300]
    )

    left_eye_aspect_ratio = (
        left_eye_width / left_eye_height
        if left_eye_height != 0 else 0
    )

    right_eye_aspect_ratio = (
        right_eye_width / right_eye_height
        if right_eye_height != 0 else 0
    )


    face_aspect_ratio = (
        face_width / face_height
        if face_height != 0 else 0
    )

    nose_width_ratio = (
        nose_width / face_width
        if face_width != 0 else 0
    )

    nose_mouth_ratio = (
        nose_to_mouth / face_height
        if face_height != 0 else 0
    )

    eye_width_symmetry = symmetry(
        left_eye_width,
        right_eye_width
    )

    eye_height_symmetry = symmetry(
        left_eye_height,
        right_eye_height
    )

    eye_aspect_symmetry = symmetry(
        left_eye_aspect_ratio,
        right_eye_aspect_ratio
    )

    eyebrow_symmetry = symmetry(
        distance(landmarks[70], landmarks[10]),
        distance(landmarks[300], landmarks[10])
    )

    nose_angle = angle(
        landmarks[129],
        landmarks[1],
        landmarks[358]
    )

    return {
        "left_eye_width": left_eye_width,
        "right_eye_width": right_eye_width,
        "left_eye_height": left_eye_height,
        "right_eye_height": right_eye_height,
        "eye_distance": eye_distance,
        "mouth_width": mouth_width,
        "mouth_height": mouth_height,
        "face_width": face_width,
        "face_height": face_height,
        "nose_width": nose_width,
        "nose_to_mouth": nose_to_mouth,
        "eyebrow_distance": eyebrow_distance,
        "left_eye_aspect_ratio": left_eye_aspect_ratio,
        "right_eye_aspect_ratio": right_eye_aspect_ratio,
        "face_aspect_ratio": face_aspect_ratio,
        "nose_width_ratio": nose_width_ratio,
        "nose_mouth_ratio": nose_mouth_ratio,
        "eye_width_symmetry": eye_width_symmetry,
        "eye_height_symmetry": eye_height_symmetry,
        "eye_aspect_symmetry": eye_aspect_symmetry,
        "eyebrow_symmetry": eyebrow_symmetry,
        "nose_angle": nose_angle
    }


# ==============================
# DETECT LANDMARKS
# ==============================

if not JSON_OUTPUT:
    print(f"Processing image: {IMAGE_PATH}")

try:
    image = mp.Image.create_from_file(IMAGE_PATH)
    result = detector.detect(image)

    if not result.face_landmarks:
        if JSON_OUTPUT:
            print(json.dumps({
                "status": "error",
                "message": "No face detected in the image."
            }))
            sys.exit(0)
        else:
            print("No face detected in the image.")
            sys.exit(1)

    landmarks = result.face_landmarks[0]

    if not JSON_OUTPUT:
        print(f"Detected {len(landmarks)} facial landmarks.")

    # ==============================
    # EXTRACT FEATURES
    # ==============================

    features = extract_features(landmarks)

    # ==============================
    # PREPARE MODEL INPUT
    # ==============================

    feature_order = [
        "left_eye_width",
        "right_eye_width",
        "left_eye_height",
        "right_eye_height",
        "eye_distance",
        "mouth_width",
        "mouth_height",
        "face_width",
        "face_height",
        "nose_width",
        "nose_to_mouth",
        "eyebrow_distance",
        "left_eye_aspect_ratio",
        "right_eye_aspect_ratio",
        "face_aspect_ratio",
        "nose_width_ratio",
        "nose_mouth_ratio",
        "eye_width_symmetry",
        "eye_height_symmetry",
        "eye_aspect_symmetry",
        "eyebrow_symmetry",
        "nose_angle"
    ]

    X = pd.DataFrame(
        [features],
        columns=feature_order
    )

    # ==============================
    # PREDICTION
    # ==============================

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    class_names = model.classes_

    probability_map = dict(
        zip(class_names, probabilities)
    )

    confidence = probability_map[prediction] * 100

    if JSON_OUTPUT:
        # Load importance file if available
        importance_path = BASE_DIR / "processed" / "feature_importance.csv"
        top_features = []
        if importance_path.exists():
            imp_df = pd.read_csv(importance_path)
            for _, r in imp_df.head(10).iterrows():
                f_name = r["feature"]
                top_features.append({
                    "feature": f_name,
                    "importance": float(r["importance"]),
                    "value": float(features.get(f_name, 0.0))
                })

        output_payload = {
            "status": "success",
            "prediction": prediction,
            "confidence": round(float(confidence), 2),
            "probabilities": {k: round(float(v) * 100, 2) for k, v in probability_map.items()},
            "landmarks_count": len(landmarks),
            "features": {k: float(v) for k, v in features.items()},
            "top_features": top_features,
            "symmetry_metrics": {
                "eye_width_symmetry": float(features["eye_width_symmetry"]),
                "eye_height_symmetry": float(features["eye_height_symmetry"]),
                "eye_aspect_symmetry": float(features["eye_aspect_symmetry"]),
                "eyebrow_symmetry": float(features["eyebrow_symmetry"])
            },
            "geometry_metrics": {
                "face_aspect_ratio": float(features["face_aspect_ratio"]),
                "left_eye_aspect_ratio": float(features["left_eye_aspect_ratio"]),
                "right_eye_aspect_ratio": float(features["right_eye_aspect_ratio"]),
                "nose_width_ratio": float(features["nose_width_ratio"]),
                "nose_mouth_ratio": float(features["nose_mouth_ratio"])
            }
        }
        print(json.dumps(output_payload))
    else:
        # ==============================
        # RESULT
        # ==============================

        print()
        print("==============================")
        print("DEEPTRACE PREDICTION")
        print("==============================")

        print(f"Prediction : {prediction}")
        print(f"Confidence : {confidence:.2f}%")

        print()
        print("Class probabilities:")

        for class_name in class_names:
            print(
                f"{class_name:6} : "
                f"{probability_map[class_name] * 100:.2f}%"
            )

        print()
        print("==============================")
        print("EXTRACTED FEATURES")
        print("==============================")

        for name in feature_order:
            print(
                f"{name:30} : "
                f"{features[name]:.6f}"
            )

        print()
        print("Prediction completed successfully.")

finally:
    detector.close()