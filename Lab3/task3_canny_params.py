import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# Lab 03 - Task 3: Parameter Analysis of Canny Edge Detection
# Self-contained Colab cell: downloads dataset, tests threshold and
# kernel-size combinations, builds Table 2, and picks the best config.
# =====================================================================

# ---- 1. Dataset (re-download in case runtime was reset) ----
!pip install kagglehub -q
import kagglehub
path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
base = os.path.join(path, "Skin cancer ISIC The International Skin Imaging Collaboration")
DATASET_DIR = os.path.join(base, "Train")

OUTPUT_DIR = "/content/outputs"
IMG_SIZE = (256, 256)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# use the same representative image as Task 1 / Task 2 for consistency
first_class = sorted(os.listdir(DATASET_DIR))[0]
cls_dir = os.path.join(DATASET_DIR, first_class)
img_file = [f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))][0]
img_path = os.path.join(cls_dir, img_file)
print("Using image:", img_path)

img = cv2.imread(img_path)
img = cv2.resize(img, IMG_SIZE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_eq = cv2.equalizeHist(gray)   # same contrast fix used in Task 1/2

# ---- 2. Canny with configurable threshold + Gaussian kernel size ----
def canny_with_config(gray_img, low, high, ksize):
    """
    ksize = Gaussian smoothing kernel applied BEFORE Canny
    (Canny itself always uses a fixed 3x3 Sobel internally; varying the
    pre-smoothing kernel is the standard way to study 'kernel size' effect
    on Canny's sensitivity to fine detail / noise).
    """
    blurred = cv2.GaussianBlur(gray_img, (ksize, ksize), 0)
    return cv2.Canny(blurred, low, high)

def edge_pixel_count(edge_img):
    return int(np.sum(edge_img > 0))

def edge_density(edge_img):
    return 100.0 * np.sum(edge_img > 0) / edge_img.size

def quality_label(density):
    # heuristic bands for a dermoscopy lesion image (empirically tuned)
    if density < 2:
        return "Low (under-detected)"
    elif density < 6:
        return "Good"
    elif density < 12:
        return "Slightly noisy"
    else:
        return "Over-detected (noisy)"

# ---- 3. Define the required configurations ----
configs = [
    {"name": "Canny-1", "low": 30,  "high": 100, "ksize": 3},
    {"name": "Canny-2", "low": 50,  "high": 150, "ksize": 3},
    {"name": "Canny-3", "low": 100, "high": 200, "ksize": 3},
    {"name": "Canny-4", "low": 50,  "high": 150, "ksize": 5},  # kernel-size variant
]

rows = []
results = {}
for cfg in configs:
    edges = canny_with_config(gray_eq, cfg["low"], cfg["high"], cfg["ksize"])
    results[cfg["name"]] = edges
    density = edge_density(edges)
    count = edge_pixel_count(edges)
    quality = quality_label(density)

    if density < 2:
        obs = "Thresholds too strict for this image - real lesion boundaries missed."
    elif density > 12:
        obs = "Thresholds too loose - texture/noise picked up as false edges."
    else:
        obs = "Reasonable balance between detecting real edges and suppressing noise."

    rows.append({
        "Configuration": cfg["name"],
        "Low Threshold": cfg["low"],
        "High Threshold": cfg["high"],
        "Kernel Size": f'{cfg["ksize"]}x{cfg["ksize"]}',
        "Edge Quality": quality,
        "Number of Detected Edges (px)": count,
        "Observation": obs,
    })

table2 = pd.DataFrame(rows)
print("\n=== Table 2: Canny Parameter Analysis ===")
print(table2.to_string(index=False))
table2.to_csv(os.path.join(OUTPUT_DIR, "table2_canny_params.csv"), index=False)
print(f"\nSaved: {OUTPUT_DIR}/table2_canny_params.csv")

# ---- 4. Pick the best configuration ----
# "Best" = closest edge density to the sweet spot (here: 4-6%), which
# balances catching real lesion-boundary edges without noise clutter.
target_density = 5.0
table2["_density"] = [edge_density(results[n]) for n in table2["Configuration"]]
table2["_score"] = (table2["_density"] - target_density).abs()
best_row = table2.loc[table2["_score"].idxmin()]
best_name = best_row["Configuration"]
print(f"\nSelected best configuration: {best_name} "
      f"(Low={best_row['Low Threshold']}, High={best_row['High Threshold']}, "
      f"Kernel={best_row['Kernel Size']}) "
      f"-> most useful edge representation for this dataset.")

# ---- 5. Visual comparison of all 4 configurations ----
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original", fontsize=11)
axes[0].axis("off")

for i, cfg in enumerate(configs):
    ax = axes[i + 1]
    edges = results[cfg["name"]]
    label = f'{cfg["name"]}\nL={cfg["low"]}, H={cfg["high"]}, k={cfg["ksize"]}x{cfg["ksize"]}'
    if cfg["name"] == best_name:
        label += "\n(BEST)"
    ax.imshow(edges, cmap="gray")
    ax.set_title(label, fontsize=10,
                 color="green" if cfg["name"] == best_name else "black",
                 fontweight="bold" if cfg["name"] == best_name else "normal")
    ax.axis("off")

fig.suptitle(f"Task 3 - Canny Parameter Analysis | Class: {first_class}", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_fig = os.path.join(OUTPUT_DIR, "task3_canny_param_analysis.png")
plt.savefig(out_fig, dpi=150)
plt.show()
print(f"Saved: {out_fig}")

table2 = table2.drop(columns=["_density", "_score"])
