import pandas as pd
import numpy as np

FEATURE_FILE = "processed/features.csv"

print("Loading feature dataset...")

df = pd.read_csv(FEATURE_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())

features = df.drop(columns=["label", "dataset_index"])

print("\nChecking for NaN values...")
print(features.isna().sum().sum())

print("\nChecking for infinite values...")
print(np.isinf(features.to_numpy()).sum())

print("\nFeature statistics:")
print(
    features.describe()
    .T[
        [
            "min",
            "max",
            "mean",
            "std"
        ]
    ]
    .to_string()
)

print("\nPotentially extreme features:")

for column in features.columns:

    maximum = features[column].max()
    minimum = features[column].min()

    if not np.isfinite(maximum) or not np.isfinite(minimum):
        print(f"{column}: contains non-finite values")

print("\nFeatures with very low variation:")

for column in features.columns:

    if features[column].std() < 0.000001:
        print(
            f"{column}: std={features[column].std():.10f}"
        )

print("\nFeature quality check completed.")