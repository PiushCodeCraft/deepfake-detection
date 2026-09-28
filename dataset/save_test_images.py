from datasets import load_dataset
import os

print("Loading dataset...")

ds = load_dataset("Hemg/deepfake-and-real-images")
train = ds["train"]

label_names = train.features["label"].names

fake_saved = False
real_saved = False

os.makedirs("test_images", exist_ok=True)

for i in range(len(train)):

    label = label_names[train[i]["label"]]

    if label == "Fake" and not fake_saved:
        train[i]["image"].save("test_images/fake_original.jpg")
        fake_saved = True
        print("Saved Fake image.")

    elif label == "Real" and not real_saved:
        train[i]["image"].save("test_images/real_original.jpg")
        real_saved = True
        print("Saved Real image.")

    if fake_saved and real_saved:
        break

print("Test images created.")