# pyrefly: ignore [missing-import]
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from mediapipe.tasks import python
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python import vision
# pyrefly: ignore [missing-import]
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
# 3. Create MediaPipe detector
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
# 4. Distance between two landmarks
# --------------------------------------------------

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# --------------------------------------------------
# 5. Calculate angle between three landmarks
# --------------------------------------------------

def angle(a, b, c):

    vector1 = np.array([
        a.x - b.x,
        a.y - b.y
    ])

    vector2 = np.array([
        c.x - b.x,
        c.y - b.y
    ])

    cosine = np.dot(vector1, vector2)

    denominator = (
        np.linalg.norm(vector1) *
        np.linalg.norm(vector2)
    )

    if denominator == 0:
        return 0.0

    cosine = cosine / denominator

    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(cosine))


# --------------------------------------------------
# 6. Extract features
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


    # ==============================================
    # IMPORTANT LANDMARKS
    # ==============================================

    # Eyes
    left_eye_outer = landmarks[33]
    left_eye_inner = landmarks[133]

    right_eye_inner = landmarks[362]
    right_eye_outer = landmarks[263]

    # Eye top/bottom
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]

    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]


    # Nose
    nose_tip = landmarks[1]
    nose_left = landmarks[129]
    nose_right = landmarks[358]


    # Mouth
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]

    mouth_top = landmarks[13]
    mouth_bottom = landmarks[14]

    # Better mouth height
    mouth_top_outer = landmarks[0]
    mouth_bottom_outer = landmarks[17]


    # Face
    face_left = landmarks[234]
    face_right = landmarks[454]

    face_top = landmarks[10]
    face_bottom = landmarks[152]


    # Eyebrows
    left_eyebrow = landmarks[70]
    right_eyebrow = landmarks[300]


    # ==============================================
    # BASIC GEOMETRY
    # ==============================================

    left_eye_width = distance(
        left_eye_outer,
        left_eye_inner
    )

    right_eye_width = distance(
        right_eye_outer,
        right_eye_inner
    )

    left_eye_height = distance(
        left_eye_top,
        left_eye_bottom
    )

    right_eye_height = distance(
        right_eye_top,
        right_eye_bottom
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
        mouth_top_outer,
        mouth_bottom_outer
    )


    face_width = distance(
        face_left,
        face_right
    )

    face_height = distance(
        face_top,
        face_bottom
    )


    nose_width = distance(
        nose_left,
        nose_right
    )

    nose_to_mouth = distance(
        nose_tip,
        mouth_top
    )


    eyebrow_distance = distance(
        left_eyebrow,
        right_eyebrow
    )


    # ==============================================
    # RATIOS
    # ==============================================

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


    # ==============================================
    # SYMMETRY FEATURES
    # ==============================================

    # Eye width symmetry
    eye_width_symmetry = (
        abs(left_eye_width - right_eye_width)
        / ((left_eye_width + right_eye_width) / 2)
        if (left_eye_width + right_eye_width) != 0
        else 0
    )


    # Eye height symmetry
    eye_height_symmetry = (
        abs(left_eye_height - right_eye_height)
        / ((left_eye_height + right_eye_height) / 2)
        if (left_eye_height + right_eye_height) != 0
        else 0
    )


    # Eye aspect ratio symmetry
    eye_aspect_symmetry = (
        abs(
            left_eye_aspect_ratio -
            right_eye_aspect_ratio
        )
        / (
            (left_eye_aspect_ratio +
             right_eye_aspect_ratio) / 2
        )
        if (
            left_eye_aspect_ratio +
            right_eye_aspect_ratio
        ) != 0
        else 0
    )


    # Eyebrow symmetry
    # Compare eyebrow positions relative to face center
    face_center_x = (
        face_left.x + face_right.x
    ) / 2

    left_eyebrow_offset = abs(
        face_center_x - left_eyebrow.x
    )

    right_eyebrow_offset = abs(
        right_eyebrow.x - face_center_x
    )

    eyebrow_symmetry = (
        abs(
            left_eyebrow_offset -
            right_eyebrow_offset
        )
        / (
            (left_eyebrow_offset +
             right_eyebrow_offset) / 2
        )
        if (
            left_eyebrow_offset +
            right_eyebrow_offset
        ) != 0
        else 0
    )


    # ==============================================
    # ANGLES
    # ==============================================

    nose_angle = angle(
        nose_left,
        nose_tip,
        nose_right
    )


    # ==============================================
    # RETURN FEATURE VECTOR
    # ==============================================

    features = {

        "left_eye_width":
            left_eye_width,

        "right_eye_width":
            right_eye_width,

        "left_eye_height":
            left_eye_height,

        "right_eye_height":
            right_eye_height,

        "eye_distance":
            eye_distance,

        "mouth_width":
            mouth_width,

        "mouth_height":
            mouth_height,

        "face_width":
            face_width,

        "face_height":
            face_height,

        "nose_width":
            nose_width,

        "nose_to_mouth":
            nose_to_mouth,

        "eyebrow_distance":
            eyebrow_distance,

        "left_eye_aspect_ratio":
            left_eye_aspect_ratio,

        "right_eye_aspect_ratio":
            right_eye_aspect_ratio,

        "mouth_aspect_ratio":
            mouth_aspect_ratio,

        "face_aspect_ratio":
            face_aspect_ratio,

        "nose_width_ratio":
            nose_width_ratio,

        "nose_mouth_ratio":
            nose_mouth_ratio,

        "eye_width_symmetry":
            eye_width_symmetry,

        "eye_height_symmetry":
            eye_height_symmetry,

        "eye_aspect_symmetry":
            eye_aspect_symmetry,

        "eyebrow_symmetry":
            eyebrow_symmetry,

        "nose_angle":
            nose_angle
    }

    return features


# --------------------------------------------------
# 7. Test Fake
# --------------------------------------------------

print("\n==============================")
print("FAKE IMAGE")
print("==============================")

fake_image = train[fake_index]["image"]

fake_features = extract_features(fake_image)

if fake_features:

    for name, value in fake_features.items():

        print(f"{name:30s}: {value:.6f}")

else:

    print("No face detected.")


# --------------------------------------------------
# 8. Test Real
# --------------------------------------------------

print("\n==============================")
print("REAL IMAGE")
print("==============================")

real_image = train[real_index]["image"]

real_features = extract_features(real_image)

if real_features:

    for name, value in real_features.items():

        print(f"{name:30s}: {value:.6f}")

else:

    print("No face detected.")


# --------------------------------------------------
# 9. Close detector
# --------------------------------------------------

detector.close()

print("\nFeature extraction test completed.")