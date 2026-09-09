from datasets import load_dataset

print("Downloading dataset...")

ds = load_dataset(
    "Hemg/deepfake-and-real-images"
)

print("Download completed!")
print(ds)