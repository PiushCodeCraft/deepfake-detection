# DeepTrace Dataset Pipeline

## Overview

This folder contains the Python scripts used for dataset loading, facial
landmark detection, feature extraction, machine learning dataset generation,
model training, prediction, and explainability.

The project uses the Hugging Face dataset:

Hemg/deepfake-and-real-images

The dataset contains two classes:
- Fake
- Real
The dataset is loaded directly using the Hugging Face datasets library.
The complete dataset does not need to be copied into this repository.

 ## Dataset Processing Flow
The complete DeepTrace Python pipeline is:
Hugging Face Dataset
        ↓
test_landmarks.py
        ↓
visualize_landmarks.py
        ↓
extract_geometry.py
        ↓
extract_features.py
        ↓
generate_feature_dataset.py
        ↓
processed/features.csv
        ↓
train_model.py
        ↓
Random Forest Model
        ↓
predict_image.py
        ↓
explain_prediction.py

Environment Setup
Open Command Prompt and navigate to the dataset directory:

Install the required Python packages:
py -m pip install -r requirements.txt

Verify the dependencies:
py -m pip check


Expected result:
No broken requirements found.

```text