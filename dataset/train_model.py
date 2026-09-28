import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import os


# ==============================
# SETTINGS
# ==============================

DATASET_FILE = "processed/features.csv"
MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================
# LOAD DATASET
# ==============================

print("Loading feature dataset...")

df = pd.read_csv(DATASET_FILE)

print(f"Dataset shape: {df.shape}")

print("\nClass distribution:")
print(df["label"].value_counts())


# ==============================
# PREPARE FEATURES
# ==============================

# Remove label and dataset index
X = df.drop(columns=["label", "dataset_index"])

y = df["label"]


print(f"\nNumber of features: {X.shape[1]}")


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==============================
# RANDOM FOREST
# ==============================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)


print("Training completed.")


# ==============================
# PREDICTIONS
# ==============================

y_pred = model.predict(X_test)


# ==============================
# EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Fake"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Fake"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Fake"
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Fake", "Real"]
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("DeepTrace - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "processed/confusion_matrix.png",
    dpi=300
)

plt.close()


# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)


print("\nTop 10 Important Features:")

print(
    importance.head(10).to_string(
        index=False
    )
)


# Save feature importance
importance.to_csv(
    "processed/feature_importance.csv",
    index=False
)


# ==============================
# SAVE MODEL
# ==============================

import joblib

model_path = os.path.join(
    MODEL_DIR,
    "deeptrace_random_forest.joblib"
)

joblib.dump(
    model,
    model_path
)


print("\nModel saved to:")
print(model_path)

print("\nTraining pipeline completed successfully.")