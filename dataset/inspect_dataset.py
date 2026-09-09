from datasets import load_dataset

print("Loading dataset...")

ds = load_dataset("Hemg/deepfake-and-real-images")

train = ds["train"]

print("\nDataset information:")
print(train)

print("\nLabel information:")
print(train.features["label"])

print("\nLabel names:")
print(train.features["label"].names)

print("\nCounting labels...")

counts = {}

for label_id in range(len(train.features["label"].names)):
    label_name = train.features["label"].names[label_id]
    counts[label_name] = 0

for item in train:
    label_id = item["label"]
    label_name = train.features["label"].names[label_id]
    counts[label_name] += 1

print("\nImage counts:")

for label, count in counts.items():
    print(f"{label}: {count}")