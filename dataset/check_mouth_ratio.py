import pandas as pd

FEATURE_FILE = "processed/features.csv"

df = pd.read_csv(FEATURE_FILE)

print("Top 20 highest mouth aspect ratios:\n")

result = (
    df[
        [
            "mouth_aspect_ratio",
            "mouth_width",
            "mouth_height",
            "label",
            "dataset_index"
        ]
    ]
    .sort_values(
        by="mouth_aspect_ratio",
        ascending=False
    )
    .head(20)
)

print(result.to_string(index=False))

print("\nClass distribution among top 20:")
print(result["label"].value_counts())