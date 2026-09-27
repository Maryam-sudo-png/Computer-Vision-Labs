import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

# =====================================================================
# Lab 03 - Task 2: Effect of Noise on Edge Detection
# Self-contained Colab cell: downloads dataset, adds noise, filters,
# runs edge detectors, builds the comparison figure AND Table 1.
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

# pick one representative image (first class, first image)
first_class = sorted(os.listdir(DATASET_DIR))[0]
cls_dir = os.path.join(DATASET_DIR, first_class)
img_file = [f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))][0]
img_path = os.path.join(cls_dir, img_file)
print("Using image:", img_path)

img = cv2.imread(img_path)
img = cv2.resize(img, IMG_SIZE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- 2. Noise functions ----
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
    noisy = image.copy()
    num_salt = int(np.ceil(amount * image.size * salt_vs_pepper))
    num_pepper = int(np.ceil(amount * image.size * (1.0 - salt_vs_pepper)))

    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

# ---- 3. Denoising filters ----
def gaussian_denoise(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def median_denoise(image, ksize=5):
    return cv2.medianBlur(image, ksize)

# ---- 4. Edge detectors (same as Task 1) ----
def apply_sobel(g):
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    return np.uint8(255*mag/(mag.max()+1e-8))

def apply_prewitt(g):
    kx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
    ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
    ix = cv2.filter2D(g.astype(np.float32), -1, kx)
    iy = cv2.filter2D(g.astype(np.float32), -1, ky)
    mag = np.sqrt(ix**2 + iy**2)
    return np.uint8(255*mag/(mag.max()+1e-8))

def apply_laplacian(g):
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=3)
    return np.uint8(255*np.abs(lap)/(np.abs(lap).max()+1e-8))

def apply_log(g, sigma=2.0):
    blurred = ndimage.gaussian_filter(g.astype(np.float32), sigma=sigma)
    log = ndimage.laplace(blurred)
    return np.uint8(255*np.abs(log)/(np.abs(log).max()+1e-8))

def apply_canny(g, low=30, high=90):
    g_eq = cv2.equalizeHist(g)
    return cv2.Canny(g_eq, low, high)

# ---- 5. Build all noisy / filtered versions ----
clean          = gray
noisy_gauss    = add_gaussian_noise(gray)
noisy_sp       = add_salt_pepper_noise(gray)
gauss_filtered = gaussian_denoise(noisy_gauss)   # Gaussian noise -> Gaussian filter
sp_filtered    = median_denoise(noisy_sp)        # Salt&Pepper noise -> Median filter

# ---- 6. Edge quality metric: % of pixels marked as edges ----
def edge_density(edge_img, thresh=30):
    return 100.0 * np.sum(edge_img > thresh) / edge_img.size

def quality_label(density):
    if density < 3:
        return "Low"
    elif density < 10:
        return "Medium"
    else:
        return "High"

# ---- 7. Compute Table 1 rows exactly as specified in the lab handout ----
rows = []

def add_row(detector, input_label, noise_type, preprocessing, edge_img, baseline_density=None):
    density = edge_density(edge_img)
    quality = quality_label(density)
    if baseline_density is not None and baseline_density > 0:
        sensitivity_pct = 100.0 * (density - baseline_density) / baseline_density
        sensitivity = f"{sensitivity_pct:+.1f}% vs clean"
    else:
        sensitivity = "baseline"
    rows.append({
        "Edge Detector": detector,
        "Input Image": input_label,
        "Noise Type": noise_type,
        "Preprocessing": preprocessing,
        "Edge Density (%)": round(density, 2),
        "Edge Quality": quality,
        "Noise Sensitivity": sensitivity,
    })

# --- Sobel rows ---
sobel_clean   = apply_sobel(clean)
sobel_ng      = apply_sobel(noisy_gauss)
sobel_nsp     = apply_sobel(noisy_sp)
sobel_ng_f    = apply_sobel(gauss_filtered)
sobel_nsp_f   = apply_sobel(sp_filtered)
base_d = edge_density(sobel_clean)
add_row("Sobel", "Original", "None", "None", sobel_clean, None)
add_row("Sobel", "Noisy", "Gaussian", "None", sobel_ng, base_d)
add_row("Sobel", "Noisy", "Salt & Pepper", "None", sobel_nsp, base_d)
add_row("Sobel", "Noisy", "Gaussian", "Gaussian Filter", sobel_ng_f, base_d)
add_row("Sobel", "Noisy", "Salt & Pepper", "Median Filter", sobel_nsp_f, base_d)

# --- Prewitt (Original only, per table) ---
prewitt_clean = apply_prewitt(clean)
add_row("Prewitt", "Original", "None", "None", prewitt_clean, None)

# --- Laplacian (Original only, per table) ---
laplacian_clean = apply_laplacian(clean)
add_row("Laplacian", "Original", "None", "None", laplacian_clean, None)

# --- LoG (Noisy+Gaussian filter, per table) ---
log_ng_f = apply_log(gauss_filtered)
add_row("LoG", "Noisy", "Gaussian", "Gaussian Filter", log_ng_f, None)

# --- Canny rows ---
canny_clean = apply_canny(clean)
canny_ng_f  = apply_canny(gauss_filtered)
canny_nsp_f = apply_canny(sp_filtered)
base_c = edge_density(canny_clean)
add_row("Canny", "Original", "None", "Built-in smoothing", canny_clean, None)
add_row("Canny", "Noisy", "Gaussian", "Gaussian Filter", canny_ng_f, base_c)
add_row("Canny", "Noisy", "Salt & Pepper", "Median Filter", canny_nsp_f, base_c)

table1 = pd.DataFrame(rows)
print("\n=== Table 1: Effect of Noise and Preprocessing on Edge Detection ===")
print(table1.to_string(index=False))
table1.to_csv(os.path.join(OUTPUT_DIR, "table1_noise_effect.csv"), index=False)
print(f"\nSaved: {OUTPUT_DIR}/table1_noise_effect.csv")

# ---- 8. Visual comparison figure (Sobel across all scenarios - most complete row) ----
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
top_titles = ["Original", "Gaussian Noise", "Salt & Pepper Noise",
              "Gaussian Noise\n+ Gaussian Filter", "Salt & Pepper Noise\n+ Median Filter"]
top_images = [clean, noisy_gauss, noisy_sp, gauss_filtered, sp_filtered]
bottom_images = [sobel_clean, sobel_ng, sobel_nsp, sobel_ng_f, sobel_nsp_f]

for i in range(5):
    axes[0, i].imshow(top_images[i], cmap="gray")
    axes[0, i].set_title(top_titles[i], fontsize=10)
    axes[0, i].axis("off")
    axes[1, i].imshow(bottom_images[i], cmap="gray")
    axes[1, i].set_title(f"Sobel Edges\n(density {edge_density(bottom_images[i]):.1f}%)", fontsize=10)
    axes[1, i].axis("off")

fig.suptitle(f"Task 2 - Effect of Noise & Preprocessing on Sobel Edge Detection | Class: {first_class}",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_fig = os.path.join(OUTPUT_DIR, "task2_noise_effect_sobel.png")
plt.savefig(out_fig, dpi=150)
plt.show()
print(f"Saved: {out_fig}")

# ---- 9. Extra comparison figure for Canny (built-in smoothing already handles noise differently) ----
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
canny_imgs = [canny_clean, canny_ng_f, canny_nsp_f]
canny_titles = ["Canny - Original\n(built-in smoothing)",
                "Canny - Gaussian noise\n+ Gaussian filter",
                "Canny - Salt & Pepper noise\n+ Median filter"]
for i in range(3):
    axes2[i].imshow(canny_imgs[i], cmap="gray")
    axes2[i].set_title(canny_titles[i], fontsize=10)
    axes2[i].axis("off")
fig2.suptitle("Task 2 - Canny Edge Detection Across Noise/Preprocessing Scenarios", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.92])
out_fig2 = os.path.join(OUTPUT_DIR, "task2_noise_effect_canny.png")
plt.savefig(out_fig2, dpi=150)
plt.show()
print(f"Saved: {out_fig2}")
