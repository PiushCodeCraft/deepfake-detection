import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# ==============================
# PATHS & CONFIG
# ==============================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATASET_FILE = BASE_DIR / "processed" / "features_test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "deeptrace_random_forest.joblib"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DEEPTRACE MODEL EVALUATION SUITE")
print("=" * 60)

# ==============================
# LOAD DATASET
# ==============================

if not DATASET_FILE.exists():
    raise FileNotFoundError(f"Dataset not found at {DATASET_FILE}")

df = pd.read_csv(DATASET_FILE)
print(f"Loaded dataset: {DATASET_FILE.name}")
print(f"Total samples : {len(df)}")
print("Class counts  :")
for label, cnt in df["label"].value_counts().items():
    print(f"  - {label:6}: {cnt}")

X = df.drop(columns=["label", "dataset_index"])
y = df["label"]
y_binary = (y == "Fake").astype(int)

feature_names = X.columns.tolist()
print(f"Total features: {len(feature_names)}")

# ==============================
# EVALUATE SAVED MODEL
# ==============================

if MODEL_PATH.exists():
    print("\n--- Evaluating Saved Production Model ---")
    saved_model = joblib.load(str(MODEL_PATH))

    # Standard 80/20 test split reproduction
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    y_test_binary = (y_test == "Fake").astype(int)

    y_pred = saved_model.predict(X_test)
    y_proba = saved_model.predict_proba(X_test)

    # Fake class index
    fake_idx = list(saved_model.classes_).index("Fake")
    y_scores = y_proba[:, fake_idx]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Fake")
    rec = recall_score(y_test, y_pred, pos_label="Fake")
    f1 = f1_score(y_test, y_pred, pos_label="Fake")
    roc_auc = roc_auc_score(y_test_binary, y_scores)

    print(f"Held-out Test Samples : {len(X_test)}")
    print(f"Accuracy              : {acc * 100:.2f}%")
    print(f"Precision (Fake)      : {prec * 100:.2f}%")
    print(f"Recall (Fake)         : {rec * 100:.2f}%")
    print(f"F1-Score (Fake)       : {f1 * 100:.2f}%")
    print(f"ROC-AUC Score         : {roc_auc:.4f}")

    # Generate ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test_binary, y_scores)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"DeepTrace RF (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    plt.title("DeepTrace - Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = OUTPUT_DIR / "roc_curve.png"
    plt.savefig(str(roc_plot_path), dpi=300)
    plt.close()
    print(f"Saved ROC Curve plot  : {roc_plot_path.name}")

# ==============================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# ==============================

print("\n--- Running 5-Fold Stratified Cross-Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = cross_validate(rf_cv, X, y_binary, cv=cv, scoring=scoring, n_jobs=-1)

cv_summary = {
    "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
    "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
    "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
    "cv_precision_std": float(np.std(cv_results["test_precision"])),
    "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
    "cv_recall_std": float(np.std(cv_results["test_recall"])),
    "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
    "cv_f1_std": float(np.std(cv_results["test_f1"])),
    "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
    "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"]))
}

print(f"5-Fold Mean Accuracy : {cv_summary['cv_accuracy_mean']*100:.2f}% (±{cv_summary['cv_accuracy_std']*100:.2f}%)")
print(f"5-Fold Mean Precision: {cv_summary['cv_precision_mean']*100:.2f}% (±{cv_summary['cv_precision_std']*100:.2f}%)")
print(f"5-Fold Mean Recall   : {cv_summary['cv_recall_mean']*100:.2f}% (±{cv_summary['cv_recall_std']*100:.2f}%)")
print(f"5-Fold Mean F1-Score : {cv_summary['cv_f1_mean']*100:.2f}% (±{cv_summary['cv_f1_std']*100:.2f}%)")
print(f"5-Fold Mean ROC-AUC  : {cv_summary['cv_roc_auc_mean']:.4f} (±{cv_summary['cv_roc_auc_std']:.4f})")

# Save complete metrics report to JSON
report_payload = {
    "dataset": "Hemg/deepfake-and-real-images",
    "total_samples": len(df),
    "features_count": len(feature_names),
    "features": feature_names,
    "single_test_split": {
        "test_size": len(X_test),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc)
    },
    "cross_validation_5fold": cv_summary
}

metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(report_payload, f, indent=2)

print(f"\nSaved metrics summary to: {metrics_path.name}")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
