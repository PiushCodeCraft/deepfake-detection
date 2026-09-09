from datasets import load_dataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np

print("Loading dataset...")

ds = load_dataset("Hemg/deepfake-and-real-images")
train = ds["train"]

label_names = train.features["label"].names

# Find one Fake and one Real image
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

# Path to MediaPipe model
model_path = "../models/face_landmarker.task"

# Create MediaPipe Face Landmarker
base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# Test Fake and Real
for index, label in [
    (fake_index, "Fake"),
    (real_index, "Real")
]:

    print(f"\nTesting {label} image...")

    image = train[index]["image"]

    # Convert PIL image to NumPy
    image_np = np.array(image)

    # Convert NumPy array to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    # Detect face landmarks
    result = detector.detect(mp_image)

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        print(f"{label}: Face detected!")
        print(f"Number of landmarks: {len(landmarks)}")

        # Print first landmark
        first = landmarks[0]

        print(
            f"First landmark: "
            f"x={first.x:.4f}, "
            f"y={first.y:.4f}, "
            f"z={first.z:.4f}"
        )

    else:
        print(f"{label}: No face detected.")

detector.close()

print("\nLandmark test completed.")