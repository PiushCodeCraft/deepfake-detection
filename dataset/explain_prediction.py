from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

import numpy as np
import pandas as pd
import joblib
import sys
import os


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "../models/deeptrace_random_forest.joblib"
LANDMARK_MODEL = "../models/face_landmarker.task"

DATASET_FILE = "processed/features.csv"
IMPORTANCE_FILE = "processed/feature_importance.csv"


# ==========================================
# CHECK INPUT
# ==========================================

if len(sys.argv) < 2:
    print("Usage:")
    print("py explain_prediction.py <image_path>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found: {IMAGE_PATH}")
    sys.exit(1)


# ==========================================
# LOAD MODEL AND DATA
# ==========================================

print("Loading DeepTrace model...")

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATASET_FILE)

importance_df = pd.read_csv(IMPORTANCE_FILE)

print("Model and feature data loaded.")


# ==========================================
# FEATURE ORDER
# ==========================================

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
    "mouth_aspect_ratio",
    "face_aspect_ratio",
    "nose_width_ratio",
    "nose_mouth_ratio",
    "eye_width_symmetry",
    "eye_height_symmetry",
    "eye_aspect_symmetry",
    "eyebrow_symmetry",
    "nose_angle"
]


# ==========================================
# MEDIAPIPE
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=LANDMARK_MODEL
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


# ==========================================
# HELPER FUNCTIONS
# ==========================================

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


# ==========================================
# FEATURE EXTRACTION
# ==========================================

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
        landmarks[13],
        landmarks[14]
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

    mouth_aspect_ratio = (
        mouth_width / mouth_height
        if mouth_height != 0 else 0
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
        "mouth_aspect_ratio": mouth_aspect_ratio,
        "face_aspect_ratio": face_aspect_ratio,
        "nose_width_ratio": nose_width_ratio,
        "nose_mouth_ratio": nose_mouth_ratio,
        "eye_width_symmetry": eye_width_symmetry,
        "eye_height_symmetry": eye_height_symmetry,
        "eye_aspect_symmetry": eye_aspect_symmetry,
        "eyebrow_symmetry": eyebrow_symmetry,
        "nose_angle": nose_angle
    }


# ==========================================
# PROCESS IMAGE
# ==========================================

print(f"Processing: {IMAGE_PATH}")

image = mp.Image.create_from_file(
    IMAGE_PATH
)

result = detector.detect(image)

if not result.face_landmarks:

    print("No face detected.")

    detector.close()

    sys.exit(1)


landmarks = result.face_landmarks[0]

print(f"Detected {len(landmarks)} facial landmarks.")


# ==========================================
# EXTRACT FEATURES
# ==========================================

features = extract_features(landmarks)

X = pd.DataFrame(
    [features],
    columns=feature_order
)


# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

class_names = model.classes_

probability_map = dict(
    zip(class_names, probabilities)
)

confidence = probability_map[prediction] * 100


# ==========================================
# EXPLANATION
# ==========================================

print()
print("==============================")
print("DEEPTRACE EXPLANATION")
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


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print()
print("==============================")
print("TOP IMPORTANT FEATURES")
print("==============================")


top_features = importance_df.head(10)

for _, row in top_features.iterrows():

    feature = row["feature"]
    importance = row["importance"]
    value = features[feature]

    print(
        f"{feature:30} "
        f"importance={importance:.4f} "
        f"value={value:.6f}"
    )


# ==========================================
# SYMMETRY FEATURES
# ==========================================

print()
print("==============================")
print("FACIAL SYMMETRY FEATURES")
print("==============================")

symmetry_features = [
    "eye_width_symmetry",
    "eye_height_symmetry",
    "eye_aspect_symmetry",
    "eyebrow_symmetry"
]

for feature in symmetry_features:

    print(
        f"{feature:25} : "
        f"{features[feature]:.6f}"
    )


# ==========================================
# GEOMETRY FEATURES
# ==========================================

print()
print("==============================")
print("KEY GEOMETRY FEATURES")
print("==============================")

geometry_features = [
    "left_eye_aspect_ratio",
    "right_eye_aspect_ratio",
    "nose_width",
    "face_height",
    "mouth_width",
    "nose_to_mouth",
    "eyebrow_distance",
    "face_aspect_ratio"
]

for feature in geometry_features:

    print(
        f"{feature:30} : "
        f"{features[feature]:.6f}"
    )


detector.close()

print()
print("Explanation generated successfully.")