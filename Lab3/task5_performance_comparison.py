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

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# =====================================================================
# Lab 03 - Task 5: Classification Performance Comparison (Table 3)
# Self-contained: re-downloads dataset, rebuilds Set A/B/C, retrains
# SVM / Random Forest / KNN, and additionally trains 2 CNN models,
# then assembles the final Table 3.
#
# ASSUMPTIONS (edit if they don't match your earlier labs/tasks):
#   - "Best filter" for Set B = Gaussian filtering (from Task 4 default)
#   - "Best Canny config" for Set C = low=50, high=150, kernel=5x5 (Task 3 default)
#   - Table 3's single Precision/Recall/F1/Training/Inference columns
#     are reported using each model's performance on the EDGE dataset
#     (Set C), since that's this lab's focus - Accuracy Raw/Filtered/Edge
#     columns still show all three for comparison.
# =====================================================================

# ---- 0. Config ----
BEST_FILTER = "gaussian"
BEST_CANNY_LOW, BEST_CANNY_HIGH, BEST_CANNY_KSIZE = 50, 150, 5

IMG_SIZE = (64, 64)
SAMPLES_PER_CLASS = 60
RANDOM_STATE = 42
CNN_EPOCHS = 10
CNN_BATCH_SIZE = 32

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
print("Classes:", classes)

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

# ---- 3. Same split (indices) for everything ----
idx = np.arange(len(y))
idx_train, idx_temp = train_test_split(idx, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, stratify=y[idx_temp], random_state=RANDOM_STATE)
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
print(f"Split -> train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

image_sets = {"Raw (Set A)": set_A, "Filtered (Set B)": set_B, "Edge (Set C)": set_C}

# ---- 4. Classical ML (SVM, RF, KNN) on flattened features ----
def flatten(images):
    return images.reshape(images.shape[0], -1).astype(np.float32)

def make_classifiers():
    return {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

results = []

for set_name, imgs in image_sets.items():
    X = flatten(imgs)
    Xtr, Xte = X[idx_train], X[idx_test]
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)

    for clf_name, clf in make_classifiers().items():
        t0 = time.time(); clf.fit(Xtr_s, y_train); train_time = time.time() - t0
        t0 = time.time(); y_pred = clf.predict(Xte_s)
        infer_ms = (time.time() - t0) * 1000 / len(Xte_s)

        results.append({
            "Model": clf_name, "Dataset": set_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Training Time (s)": train_time,
            "Inference Time (ms)": infer_ms,
        })
        print(f"[{clf_name:14s} | {set_name:16s}] acc={results[-1]['Accuracy']:.3f}")

# ---- 5. CNN Model 1 (shallow) and CNN Model 2 (deeper, with dropout) ----
def build_cnn1(input_shape, n_classes):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def build_cnn2(input_shape, n_classes):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

cnn_builders = {"CNN Model 1": build_cnn1, "CNN Model 2": build_cnn2}
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)

for set_name, imgs in image_sets.items():
    X = imgs.astype(np.float32) / 255.0
    X = X[..., np.newaxis]
    Xtr, Xval, Xte = X[idx_train], X[idx_val], X[idx_test]
    ytr_cat = to_categorical(y_train, num_classes)
    yval_cat = to_categorical(y_val, num_classes)

    for cnn_name, builder in cnn_builders.items():
        tf.random.set_seed(RANDOM_STATE)
        model = builder(input_shape, num_classes)

        t0 = time.time()
        model.fit(Xtr, ytr_cat, validation_data=(Xval, yval_cat),
                  epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE, verbose=0)
        train_time = time.time() - t0

        t0 = time.time()
        y_prob = model.predict(Xte, verbose=0)
        infer_ms = (time.time() - t0) * 1000 / len(Xte)
        y_pred = np.argmax(y_prob, axis=1)

        results.append({
            "Model": cnn_name, "Dataset": set_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Training Time (s)": train_time,
            "Inference Time (ms)": infer_ms,
        })
        print(f"[{cnn_name:14s} | {set_name:16s}] acc={results[-1]['Accuracy']:.3f} "
              f"(trained {CNN_EPOCHS} epochs in {train_time:.1f}s)")

full_results = pd.DataFrame(results)
full_results.to_csv(os.path.join(OUTPUT_DIR, "task5_full_results_long.csv"), index=False)

# ---- 6. Assemble Table 3 (Cross-Lab Classification Performance Comparison) ----
acc_pivot = full_results.pivot(index="Model", columns="Dataset", values="Accuracy")
acc_pivot = acc_pivot[["Raw (Set A)", "Filtered (Set B)", "Edge (Set C)"]]
acc_pivot.columns = ["Accuracy Raw (Lab 1)", "Accuracy Filtered (Lab 2)", "Accuracy Edge (Lab 3)"]

# Precision/Recall/F1/Times reported from the Edge (Set C) results, per this lab's focus
edge_metrics = full_results[full_results["Dataset"] == "Edge (Set C)"].set_index("Model")
edge_metrics = edge_metrics[["Precision", "Recall", "F1-Score", "Training Time (s)", "Inference Time (ms)"]]

table3 = acc_pivot.join(edge_metrics)
model_order = ["SVM", "Random Forest", "KNN", "CNN Model 1", "CNN Model 2"]
table3 = table3.reindex(model_order)
table3 = table3.round(4)

print("\n=== Table 3: Cross-Lab Classification Performance Comparison ===")
print(table3.to_string())
out_csv = os.path.join(OUTPUT_DIR, "table3_cross_lab_comparison.csv")
table3.to_csv(out_csv)
print(f"\nSaved: {out_csv}")

# ---- 7. Bar chart: accuracy across the three representations, per model ----
ax = acc_pivot.plot(kind="bar", figsize=(10, 5))
ax.set_ylabel("Accuracy")
ax.set_title("Task 5 - Accuracy Comparison: Raw vs Filtered vs Edge (all 5 models)")
plt.xticks(rotation=20)
plt.tight_layout()
out_fig = os.path.join(OUTPUT_DIR, "task5_accuracy_all_models.png")
plt.savefig(out_fig, dpi=150)
plt.show()
print(f"Saved: {out_fig}")

# keep these around for Task 6 (confusion matrices need the best model + its predictions)
print("\nBest model overall (by Edge accuracy):",
      acc_pivot["Accuracy Edge (Lab 3)"].idxmax())
