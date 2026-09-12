from datasets import load_dataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
import math


print("Loading dataset...")

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

ds = load_dataset("Hemg/deepfake-and-real-images")
train = ds["train"]

label_names = train.features["label"].names


# --------------------------------------------------
# 2. Find one Fake and one Real image
# --------------------------------------------------

fake_index = None
real_index = None

for i in range(len(train)):

    label = label_names[train[i]["label"]]

    if label == "Fake" and fake_index is None:
        fake_index = i

    if label == "Real" and real_index is None:
        real_index = i

    if fake_index is not None and real_index is not None:
        break


print(f"Fake index: {fake_index}")
print(f"Real index: {real_index}")


# --------------------------------------------------
# 3. Load MediaPipe Face Landmarker
# --------------------------------------------------

model_path = "../models/face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------
# 4. Distance function
# --------------------------------------------------

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


# --------------------------------------------------
# 5. Geometry feature extraction
# --------------------------------------------------

def extract_features(image):

    image_np = np.array(image)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    result = detector.detect(mp_image)

    if not result.face_landmarks:

        return None

    landmarks = result.face_landmarks[0]

    # ----------------------------------------------
    # Selected MediaPipe landmark points
    # ----------------------------------------------

    # Eyes
    left_eye_outer = landmarks[33]
    left_eye_inner = landmarks[133]

    right_eye_inner = landmarks[362]
    right_eye_outer = landmarks[263]

    # Nose
    nose = landmarks[1]

    # Mouth
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]

    mouth_top = landmarks[13]
    mouth_bottom = landmarks[14]

    # Face boundaries
    face_left = landmarks[234]
    face_right = landmarks[454]

    face_top = landmarks[10]
    face_bottom = landmarks[152]

    # ----------------------------------------------
    # Calculate distances
    # ----------------------------------------------

    left_eye_width = distance(
        left_eye_outer,
        left_eye_inner
    )

    right_eye_width = distance(
        right_eye_outer,
        right_eye_inner
    )

    eye_distance = distance(
        left_eye_inner,
        right_eye_inner
    )

    mouth_width = distance(
        mouth_left,
        mouth_right
    )

    mouth_height = distance(
        mouth_top,
        mouth_bottom
    )

    face_width = distance(
        face_left,
        face_right
    )

    face_height = distance(
        face_top,
        face_bottom
    )

    nose_to_mouth = distance(
        nose,
        mouth_top
    )

    # ----------------------------------------------
    # Ratios
    # ----------------------------------------------

    eye_width_ratio = (
        left_eye_width / right_eye_width
        if right_eye_width != 0 else 0
    )

    mouth_aspect_ratio = (
        mouth_width / mouth_height
        if mouth_height != 0 else 0
    )

    face_aspect_ratio = (
        face_width / face_height
        if face_height != 0 else 0
    )

    nose_mouth_ratio = (
        nose_to_mouth / face_height
        if face_height != 0 else 0
    )

    # ----------------------------------------------
    # Return features
    # ----------------------------------------------

    features = {

        "left_eye_width": left_eye_width,

        "right_eye_width": right_eye_width,

        "eye_distance": eye_distance,

        "mouth_width": mouth_width,

        "mouth_height": mouth_height,

        "face_width": face_width,

        "face_height": face_height,

        "nose_to_mouth": nose_to_mouth,

        "eye_width_ratio": eye_width_ratio,

        "mouth_aspect_ratio": mouth_aspect_ratio,

        "face_aspect_ratio": face_aspect_ratio,

        "nose_mouth_ratio": nose_mouth_ratio
    }

    return features


# --------------------------------------------------
# 6. Test Fake image
# --------------------------------------------------

print("\n==============================")
print("FAKE IMAGE GEOMETRY")
print("==============================")

fake_image = train[fake_index]["image"]

fake_features = extract_features(fake_image)

if fake_features:

    for name, value in fake_features.items():

        print(
            f"{name:25s}: {value:.6f}"
        )

else:

    print("No face detected in Fake image.")


# --------------------------------------------------
# 7. Test Real image
# --------------------------------------------------

print("\n==============================")
print("REAL IMAGE GEOMETRY")
print("==============================")

real_image = train[real_index]["image"]

real_features = extract_features(real_image)

if real_features:

    for name, value in real_features.items():

        print(
            f"{name:25s}: {value:.6f}"
        )

else:

    print("No face detected in Real image.")


# --------------------------------------------------
# 8. Close detector
# --------------------------------------------------

detector.close()

print("\nGeometry feature extraction test completed.")