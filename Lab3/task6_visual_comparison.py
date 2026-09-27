import os, cv2, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

# =====================================================================
# Lab 03 - Task 6: Visual Comparison of Classification Results
# Self-contained: rebuilds Set A/B/C, trains the BEST model from
# Task 5 (Random Forest), plots confusion matrices for all three
# representations, and a bar chart comparing Accuracy/Precision/
# Recall/F1 across them.
#
# If your Task 5 run picked a different "best model", change
# BEST_MODEL_NAME below (options: "SVM", "Random Forest", "KNN").
# =====================================================================

BEST_MODEL_NAME = "Random Forest"   # <-- from Task 5's printed "Best model overall"

BEST_FILTER = "gaussian"
BEST_CANNY_LOW, BEST_CANNY_HIGH, BEST_CANNY_KSIZE = 50, 150, 5
IMG_SIZE = (64, 64)
SAMPLES_PER_CLASS = 60
RANDOM_STATE = 42

# ---- 1. Dataset ----
!pip install kagglehub -q
import kagglehub
path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
base = os.path.join(path, "Skin cancer ISIC The International Skin Imaging Collaboration")
DATASET_DIR = os.path.join(base, "Train")
OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = sorted(d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, d)))
num_classes = len(classes)

def load_dataset(dataset_dir, classes, samples_per_class, img_size):
    images, labels = [], []
    for label_idx, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))][:samples_per_class]
        for f in files:
            img = cv2.imread(os.path.join(cls_dir, f))
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray)
            labels.append(label_idx)
    return np.array(images), np.array(labels)

print("Loading images...")
raw_images, labels = load_dataset(DATASET_DIR, classes, SAMPLES_PER_CLASS, IMG_SIZE)
print(f"Loaded {len(raw_images)} images across {num_classes} classes.")

# ---- 2. Build Set A / B / C ----
def build_filtered(images, method="gaussian"):
    return np.array([cv2.GaussianBlur(g, (5, 5), 0) if method == "gaussian"
                      else cv2.medianBlur(g, 5) for g in images])

def build_edges(images, low, high, ksize):
    out = []
    for g in images:
        g_eq = cv2.equalizeHist(g)
        blurred = cv2.GaussianBlur(g_eq, (ksize, ksize), 0)
        out.append(cv2.Canny(blurred, low, high))
    return np.array(out)

set_A = raw_images
set_B = build_filtered(raw_images, BEST_FILTER)
set_C = build_edges(raw_images, BEST_CANNY_LOW, BEST_CANNY_HIGH, BEST_CANNY_KSIZE)
y = labels

image_sets = {"Raw (Set A)": set_A, "Filtered (Set B)": set_B, "Edge (Set C)": set_C}

# ---- 3. Same split for everything ----
idx = np.arange(len(y))
idx_train, idx_temp = train_test_split(idx, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, stratify=y[idx_temp], random_state=RANDOM_STATE)
y_train, y_test = y[idx_train], y[idx_test]

def flatten(images):
    return images.reshape(images.shape[0], -1).astype(np.float32)

def make_best_model():
    # only Random Forest / SVM / KNN supported here (classical models from Task 4/5)
    if BEST_MODEL_NAME == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    elif BEST_MODEL_NAME == "SVM":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
    elif BEST_MODEL_NAME == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Unsupported BEST_MODEL_NAME for this script.")

# ---- 4. Train best model on each dataset, collect predictions + metrics ----
predictions = {}
metrics_rows = []

for set_name, imgs in image_sets.items():
    X = flatten(imgs)
    Xtr, Xte = X[idx_train], X[idx_test]
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)

    clf = make_best_model()
    clf.fit(Xtr_s, y_train)
    y_pred = clf.predict(Xte_s)
    predictions[set_name] = y_pred

    metrics_rows.append({
        "Dataset": set_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    })
    print(f"[{BEST_MODEL_NAME} | {set_name}] accuracy = {metrics_rows[-1]['Accuracy']:.3f}")

metrics_df = pd.DataFrame(metrics_rows).set_index("Dataset")
metrics_df = metrics_df.loc[["Raw (Set A)", "Filtered (Set B)", "Edge (Set C)"]]
print(f"\n=== {BEST_MODEL_NAME} - Metrics across representations ===")
print(metrics_df.round(4).to_string())
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "task6_best_model_metrics.csv"))

# ---- 5. Confusion matrices (one per dataset) ----
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
short_labels = [c[:10] for c in classes]  # shorten for readability

for ax, (set_name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=short_labels, yticklabels=short_labels, cbar=False)
    ax.set_title(f"{BEST_MODEL_NAME} - {set_name}\nAccuracy = {metrics_df.loc[set_name, 'Accuracy']:.3f}",
                 fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)

fig.suptitle(f"Task 6 - Confusion Matrices for Best Model ({BEST_MODEL_NAME})", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out_cm = os.path.join(OUTPUT_DIR, "task6_confusion_matrices.png")
plt.savefig(out_cm, dpi=150)
plt.show()
print(f"Saved: {out_cm}")

# ---- 6. Bar chart: Accuracy / Precision / Recall / F1 across representations ----
ax = metrics_df.plot(kind="bar", figsize=(9, 5))
ax.set_ylabel("Score")
ax.set_title(f"Task 6 - {BEST_MODEL_NAME}: Metric Comparison (Raw vs Filtered vs Edge)")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.tight_layout()
out_bar = os.path.join(OUTPUT_DIR, "task6_metrics_bar_chart.png")
plt.savefig(out_bar, dpi=150)
plt.show()
print(f"Saved: {out_bar}")

# ---- 7. Auto discussion summary (edit/expand this in your report) ----
raw_acc = metrics_df.loc["Raw (Set A)", "Accuracy"]
filt_acc = metrics_df.loc["Filtered (Set B)", "Accuracy"]
edge_acc = metrics_df.loc["Edge (Set C)", "Accuracy"]

print("\n=== Discussion: Does edge detection improve or reduce classification performance? ===")
if edge_acc < raw_acc and edge_acc < filt_acc:
    print(f"Edge-only images REDUCED accuracy ({edge_acc:.3f}) compared to Raw ({raw_acc:.3f}) "
          f"and Filtered ({filt_acc:.3f}). Edge maps keep only lesion boundary/shape information "
          "and discard color, texture and intensity cues - which matter a lot for distinguishing "
          "between skin lesion classes in this dataset (e.g. melanoma vs. nevus often differ more "
          "in color/texture than in outline shape). This is consistent with Question 5 in the "
          "handout: texture, color and intensity information is lost when only edges are kept.")
elif edge_acc > raw_acc and edge_acc > filt_acc:
    print(f"Edge-only images IMPROVED accuracy ({edge_acc:.3f}) over Raw ({raw_acc:.3f}) "
          f"and Filtered ({filt_acc:.3f}). This suggests the model benefited from a simplified, "
          "noise-reduced boundary representation for this dataset/classifier combination.")
else:
    print(f"Results were mixed: Raw={raw_acc:.3f}, Filtered={filt_acc:.3f}, Edge={edge_acc:.3f}. "
          "No single representation dominated across all metrics - discuss the trade-offs in your report.")
