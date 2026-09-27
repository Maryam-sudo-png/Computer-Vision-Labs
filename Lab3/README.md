# Lab 3 — Edge Detection Techniques and Their Impact on Classification Performance

Computer Vision lab exploring classical edge detection methods and evaluating whether edge-based representations help or hurt downstream image classification, using the [ISIC Skin Cancer (9 Classes)](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) dataset.

## Overview

This lab implements and compares five edge detection techniques, studies their robustness to noise, tunes Canny's parameters, and then measures how raw, filtered, and edge-only image representations affect classification accuracy across five models (SVM, Random Forest, KNN, and two CNN architectures).

## Tasks

| Task | Description |
|------|-------------|
| 1 | Comparative edge detection — Sobel (Gx/Gy/magnitude), Prewitt, Laplacian, LoG, and Canny across multiple lesion classes |
| 2 | Effect of Gaussian and Salt-&-Pepper noise on edge quality, with Gaussian/Median filtering as preprocessing |
| 3 | Canny threshold and kernel-size parameter sweep, with automatic selection of the best configuration |
| 4 | Classification on three dataset variants — Raw, Filtered, and Edge-only images — using SVM, Random Forest, and KNN |
| 5 | Full cross-model performance comparison, adding two CNN architectures, on the same representations |
| 6 | Confusion matrices and metric breakdown for the best-performing model (Random Forest) across all three representations |

## Key Finding

Across every model tested, classification accuracy followed the pattern **Raw ≥ Filtered > Edge-only**. Reducing images to edge maps discards color, texture, and intensity cues that turned out to matter more than lesion boundary shape for distinguishing between skin lesion classes in this dataset — so edge detection **reduced** classification performance here rather than improving it.

## Tech Stack

- Python, OpenCV, NumPy, SciPy (edge detection & noise/filtering)
- scikit-learn (SVM, Random Forest, KNN)
- TensorFlow / Keras (CNN models)
- pandas, matplotlib, seaborn (analysis & visualization)

## Structure

```
Lab3/
├── task1_edge_detection.py       # Comparative edge detection
├── task2_noise_effect.py         # Noise robustness + Table 1
├── task3_canny_params.py         # Canny parameter analysis + Table 2
├── task4_classification.py       # Raw/Filtered/Edge classification (SVM, RF, KNN)
├── task5_performance_comparison.py  # Full comparison incl. CNNs + Table 3
├── task6_visual_comparison.py    # Confusion matrices & metric charts
└── outputs/                      # Generated figures and result tables (CSV)
```

## Dataset

[Skin Cancer ISIC — 9 Classes](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) (Kaggle), covering melanoma, basal cell carcinoma, nevus, dermatofibroma, and five other lesion classes.

## Author

Maryam — BS Artificial Intelligence, COMSATS University Islamabad (Wah Campus)
