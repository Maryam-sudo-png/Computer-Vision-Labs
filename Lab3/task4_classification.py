import os, cv2, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# =====================================================================
# Lab 03 - Task 4: Classification Using Edge Maps
# Self-contained Colab cell.
#
# ASSUMPTION (please adjust if it doesn't match your Lab 02 result):
#   "Best filtering method" for Set B = Gaussian filtering.
#   Change BEST_FILTER below to "median" if that was your Lab 02 result.
#
# "Best edge detection config" for Set C = the winning Canny config
#   from Task 3 (defaults to low=50, high=150, ksize=5 - update the
#   BEST_CANNY_* constants below if Task 3 picked a different one on
#   your run).
# =====================================================================

# ---- 0. Config / assumptions (EDIT THESE if your Task 3 best differs) ----
BEST_FILTER = "gaussian"          # "gaussian" or "median" - from Lab 02
BEST_CANNY_LOW = 50
BEST_CANNY_HIGH = 150
BEST_CANNY_KSIZE = 5

IMG_SIZE = (64, 64)                # smaller size keeps classical ML fast
SAMPLES_PER_CLASS = 60             # cap per class so this runs in a few minutes
                                    # (raise this later for a fuller run)
RANDOM_STATE = 42

# ---- 1. Dataset (re-download in case runtime was reset) ----
!pip install kagglehub -q
import kagglehub
path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
base = os.path.join(path, "Skin cancer ISIC The International Skin Imaging Collaboration")
DATASET_DIR = os.path.join(base, "Train")
OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = sorted(d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, d)))
print("Classes:", classes)

# ---- 2. Load raw images + labels (capped per class for speed) ----
def load_dataset(dataset_dir, classes, samples_per_class, img_size):
    images, labels = [], []
    for label_idx, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        files = files[:samples_per_class]
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
print(f"Loaded {len(raw_images)} images across {len(classes)} classes.")

# ---- 3. Build Set A (raw), Set B (filtered), Set C (edge) ----
def build_filtered(images, method="gaussian"):
    out = []
    for g in images:
        if method == "gaussian":
            out.append(cv2.GaussianBlur(g, (5, 5), 0))
        else:  # median
            out.append(cv2.medianBlur(g, 5))
    return np.array(out)

def build_edges(images, low, high, ksize):
    out = []
    for g in images:
        g_eq = cv2.equalizeHist(g)
        blurred = cv2.GaussianBlur(g_eq, (ksize, ksize), 0)
        edges = cv2.Canny(blurred, low, high)
        out.append(edges)
    return np.array(out)

print("Building Set A (raw)...")
set_A = raw_images
print("Building Set B (filtered)...")
set_B = build_filtered(raw_images, method=BEST_FILTER)
print("Building Set C (edge)...")
set_C = build_edges(raw_images, BEST_CANNY_LOW, BEST_CANNY_HIGH, BEST_CANNY_KSIZE)

# flatten each image to a feature vector
def flatten(images):
    return images.reshape(images.shape[0], -1).astype(np.float32)

X_A, X_B, X_C = flatten(set_A), flatten(set_B), flatten(set_C)
y = labels

# ---- 4. SAME train/val/test split (indices) for all three sets ----
idx = np.arange(len(y))
idx_train, idx_temp = train_test_split(idx, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, stratify=y[idx_temp], random_state=RANDOM_STATE)

print(f"\nSplit sizes -> train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

def split_set(X):
    return X[idx_train], X[idx_val], X[idx_test]

datasets = {
    "Raw (Set A)": split_set(X_A),
    "Filtered (Set B)": split_set(X_B),
    "Edge (Set C)": split_set(X_C),
}
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

# ---- 5. Classifiers (same models/settings across all 3 datasets) ----
def make_classifiers():
    return {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

# ---- 6. Train + evaluate each classifier on each dataset variant ----
results = []
trained_models = {}   # keep for Task 5 (confusion matrices etc.)

for set_name, (Xtr, Xval, Xte) in datasets.items():
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    for clf_name, clf in make_classifiers().items():
        t0 = time.time()
        clf.fit(Xtr_s, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(Xte_s)
        infer_time_ms = (time.time() - t0) * 1000 / len(Xte_s)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Dataset": set_name,
            "Model": clf_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Training Time (s)": round(train_time, 2),
            "Inference Time (ms/img)": round(infer_time_ms, 3),
        })
        trained_models[(set_name, clf_name)] = (clf, scaler, y_pred)
        print(f"[{set_name:16s} | {clf_name:14s}] "
              f"acc={acc:.3f} f1={f1:.3f} train={train_time:.1f}s")

results_df = pd.DataFrame(results)
print("\n=== Task 4 Results: Raw vs Filtered vs Edge (SVM / RF / KNN) ===")
print(results_df.to_string(index=False))

out_csv = os.path.join(OUTPUT_DIR, "task4_classification_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ---- 7. Quick bar chart: accuracy per dataset per model ----
pivot = results_df.pivot(index="Model", columns="Dataset", values="Accuracy")
pivot = pivot[["Raw (Set A)", "Filtered (Set B)", "Edge (Set C)"]]
ax = pivot.plot(kind="bar", figsize=(9, 5))
ax.set_ylabel("Accuracy")
ax.set_title("Task 4 - Accuracy: Raw vs Filtered vs Edge Images")
plt.xticks(rotation=0)
plt.tight_layout()
out_fig = os.path.join(OUTPUT_DIR, "task4_accuracy_comparison.png")
plt.savefig(out_fig, dpi=150)
plt.show()
print(f"Saved: {out_fig}")

print("\nNOTE: 'trained_models' and 'y_test' / 'idx_test' are kept in memory - "
      "Task 5 will reuse them directly for confusion matrices, so keep this "
      "runtime alive before moving to Task 5.")
