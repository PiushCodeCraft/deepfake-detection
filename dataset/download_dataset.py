import kagglehub

path = kagglehub.dataset_download(
    "greatgamedota/faceforensics",
    output_dir="./raw"
)

print("Dataset downloaded to:", path)