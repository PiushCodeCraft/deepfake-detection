from datasets import load_dataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
import cv2
import os

print("Loading dataset...")

# Load dataset
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

# MediaPipe model
model_path = "../models/face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# Create output folder
output_dir = "samples"
os.makedirs(output_dir, exist_ok=True)


def process_image(index, label):
    print(f"\nProcessing {label} image...")

    image = train[index]["image"]

    # Convert PIL → NumPy
    image_np = np.array(image)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    # Detect landmarks
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        print(f"{label}: No face detected.")
        return

    landmarks = result.face_landmarks[0]

    # Convert RGB → BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    height, width = image_bgr.shape[:2]

    # Draw every landmark
    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            image_bgr,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    # Save image
    output_path = os.path.join(
        output_dir,
        f"{label.lower()}_landmarks.jpg"
    )

    cv2.imwrite(output_path, image_bgr)

    print(f"{label}: {len(landmarks)} landmarks detected.")
    print(f"Saved to: {output_path}")


# Process Fake and Real
process_image(fake_index, "Fake")
process_image(real_index, "Real")

detector.close()

print("\nLandmark visualization completed.")