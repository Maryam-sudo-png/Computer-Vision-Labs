"""
Lab 03 - Task 1: Comparative Edge Detection
Dataset: Skin Cancer ISIC (9 classes) - Kaggle
https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

This script:
  1. Loads one representative image from at least 3 different classes.
  2. Applies Sobel (Gx, Gy, magnitude), Prewitt, Laplacian, LoG, and Canny.
  3. Displays / saves a figure per class:
       Original -> Sobel -> Prewitt -> Laplacian -> LoG -> Canny

UPDATE: Canny now uses histogram equalization before thresholding.
Dermoscopy images are low-contrast, so default thresholds (100/200) on the
raw grayscale produced an all-black edge map. Equalizing contrast first
fixes this and gives proper detected edges.

HOW TO USE (Google Colab):
  1. Download the dataset (kagglehub) and set DATASET_DIR to the Train folder
     that contains one sub-folder per class, e.g.:
         DATASET_DIR = os.path.join(base, "Train")
  2. Run this script / paste this cell.
  3. Output figures are saved in OUTPUT_DIR as task1_<classname>.png
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ----------------------------------------------------------------------
# 1. CONFIG - edit these before running
# ----------------------------------------------------------------------
# In Colab, after downloading with kagglehub you already have `base`
# pointing to the dataset's root folder, so just do:
#   DATASET_DIR = os.path.join(base, "Train")
DATASET_DIR = "./data/Skin cancer ISIC The International Skin Imaging Collaboration/Train"

OUTPUT_DIR = "./outputs"
NUM_CLASSES_TO_SAMPLE = 3          # "at least three different classes"
IMG_SIZE = (256, 256)              # resize for consistent display

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 2. Option: auto-download the dataset with kagglehub instead of a manual path
# ----------------------------------------------------------------------
def download_with_kagglehub():
    import kagglehub
    path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
    print("Dataset downloaded to:", path)
    return path


# ----------------------------------------------------------------------
# 3. Helper: pick one representative image from each of N classes
# ----------------------------------------------------------------------
def get_sample_images(dataset_dir, n_classes=3):
    classes = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    if len(classes) < n_classes:
        raise ValueError(
            f"Found only {len(classes)} class folders in {dataset_dir}, "
            f"need at least {n_classes}."
        )

    samples = {}
    for cls in classes[:n_classes]:
        cls_dir = os.path.join(dataset_dir, cls)
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if not files:
            continue
        img_path = os.path.join(cls_dir, files[0])
        samples[cls] = img_path
    return samples


# ----------------------------------------------------------------------
# 4. Edge detection functions
# ----------------------------------------------------------------------
def apply_sobel(gray):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(255 * sobel_mag / (sobel_mag.max() + 1e-8))
    return (
        np.uint8(255 * np.abs(sobel_x) / (np.abs(sobel_x).max() + 1e-8)),
        np.uint8(255 * np.abs(sobel_y) / (np.abs(sobel_y).max() + 1e-8)),
        sobel_mag,
    )


def apply_prewitt(gray):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    img_x = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
    img_y = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)
    mag = np.sqrt(img_x ** 2 + img_y ** 2)
    return np.uint8(255 * mag / (mag.max() + 1e-8))


def apply_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap = np.uint8(255 * np.abs(lap) / (np.abs(lap).max() + 1e-8))
    return lap


def apply_log(gray, sigma=2.0):
    blurred = ndimage.gaussian_filter(gray.astype(np.float32), sigma=sigma)
    log = ndimage.laplace(blurred)
    log = np.uint8(255 * np.abs(log) / (np.abs(log).max() + 1e-8))
    return log


def apply_canny(gray, low=30, high=90):
    """
    NOTE: dermoscopy images are low-contrast, so raw thresholds like
    100/200 detect almost no edges (all-black output). Equalizing the
    histogram first boosts local contrast so Canny can find real edges,
    and the lower thresholds (30/90) suit the equalized image better.
    """
    gray_eq = cv2.equalizeHist(gray)
    return cv2.Canny(gray_eq, low, high)


# ----------------------------------------------------------------------
# 5. Build the required figure per class:
#    Original -> Sobel -> Prewitt -> Laplacian -> LoG -> Canny
# ----------------------------------------------------------------------
def process_and_plot(cls_name, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {img_path}, skipping.")
        return
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # mild smoothing before gradient ops

    sobel_x, sobel_y, sobel_mag = apply_sobel(gray)
    prewitt = apply_prewitt(gray)
    laplacian = apply_laplacian(gray)
    log_img = apply_log(gray)
    canny = apply_canny(gray)

    titles = ["Original", "Sobel (Gx)", "Sobel (Gy)", "Sobel Magnitude",
              "Prewitt", "Laplacian", "LoG", "Canny"]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), sobel_x, sobel_y, sobel_mag,
              prewitt, laplacian, log_img, canny]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Task 1 - Edge Detection Comparison | Class: {cls_name}", fontsize=14)
    for ax, title, im in zip(axes.ravel(), titles, images):
        cmap = None if title == "Original" else "gray"
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, f"task1_{cls_name.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.isdir(DATASET_DIR):
        print(f"[INFO] DATASET_DIR not found: {DATASET_DIR}")
        print("Edit DATASET_DIR at the top of this script to point to your "
              "downloaded dataset's Train folder, or call download_with_kagglehub().")
    else:
        samples = get_sample_images(DATASET_DIR, NUM_CLASSES_TO_SAMPLE)
        print("Using classes:", list(samples.keys()))
        for cls_name, img_path in samples.items():
            process_and_plot(cls_name, img_path)
        print("\nDone. Check the OUTPUT_DIR folder for the comparison figures "
              "(Original -> Sobel -> Prewitt -> Laplacian -> LoG -> Canny).")
