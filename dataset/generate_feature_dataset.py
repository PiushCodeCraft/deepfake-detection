from datasets import load_dataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
import csv
import os


# ==============================
# SETTINGS
# ==============================

FAKE_COUNT = 1000
REAL_COUNT = 1000

OUTPUT_DIR = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "features_test.csv")

MODEL_PATH = "../models/face_landmarker.task"


# ==============================
# LOAD DATASET
# ==============================

print("Loading dataset...")

ds = load_dataset("Hemg/deepfake-and-real-images")
train = ds["train"]

label_names = train.features["label"].names

print("Dataset loaded.")
print("Labels:", label_names)


# ==============================
# FIND FAKE AND REAL INDICES
# ==============================

fake_indices = []
real_indices = []

for i in range(len(train)):

    label = label_names[train[i]["label"]]

    if label == "Fake" and len(fake_indices) < FAKE_COUNT:
        fake_indices.append(i)

    elif label == "Real" and len(real_indices) < REAL_COUNT:
        real_indices.append(i)

    if len(fake_indices) == FAKE_COUNT and len(real_indices) == REAL_COUNT:
        break


print(f"Fake samples selected: {len(fake_indices)}")
print(f"Real samples selected: {len(real_indices)}")


# ==============================
# MEDIAPIPE SETUP
# ==============================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


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

    # Eyes
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


    # Mouth
    mouth_width = distance(
        landmarks[61],
        landmarks[291]
    )

    mouth_height = distance(
        landmarks[13],
        landmarks[14]
    )


    # Face
    face_width = distance(
        landmarks[234],
        landmarks[454]
    )

    face_height = distance(
        landmarks[10],
        landmarks[152]
    )


    # Nose
    nose_width = distance(
        landmarks[129],
        landmarks[358]
    )


    # Nose to mouth
    nose_to_mouth = distance(
        landmarks[1],
        landmarks[13]
    )


    # Eyebrow distance
    eyebrow_distance = distance(
        landmarks[70],
        landmarks[300]
    )


    # Ratios
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


    # Symmetry
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


    # Nose angle
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


# ==============================
# OUTPUT SETUP
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []

failed_images = []


# ==============================
# PROCESS IMAGES
# ==============================

samples = []

for index in fake_indices:
    samples.append((index, "Fake"))

for index in real_indices:
    samples.append((index, "Real"))


for count, (index, label) in enumerate(samples, start=1):

    print(
        f"[{count}/{len(samples)}] "
        f"Processing {label} image..."
    )

    image = train[index]["image"]

    image_np = np.array(image)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    result = detector.detect(mp_image)

    if not result.face_landmarks:

        print("  No face detected - skipped.")

        failed_images.append({
            "index": index,
            "label": label
        })

        continue


    landmarks = result.face_landmarks[0]

    features = extract_features(landmarks)

    features["label"] = label
    features["dataset_index"] = index

    rows.append(features)


# ==============================
# SAVE CSV
# ==============================

if rows:

    fieldnames = list(rows[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ==============================
# CLEANUP
# ==============================

detector.close()


print()
print("==============================")
print("FEATURE DATASET COMPLETED")
print("==============================")

print(f"Successful images : {len(rows)}")
print(f"Failed images     : {len(failed_images)}")

print(f"Saved to          : {OUTPUT_FILE}")