# Computer Vision Labs

Lab tasks for the **Computer Vision** course — BS Artificial Intelligence, COMSATS University Islamabad, Wah Campus.

Each lab lives in its own folder and gets added here as the semester progresses.

## 📂 Structure

| Lab | Topic | Status |
|-----|-------|--------|
| [Lab1](Lab1) | Transfer Learning Models, Classifier & Computational Efficiency Comparison | ✅ Done |
| [Lab2](Lab2) | Effect of Image Filtering on Skin-Lesion Classification | ✅ Done |
| [Lab3](Lab3) | Edge Detection Techniques and Their Impact on Classification Performance | ✅ Done |
| [Lab4](Lab4) | Skin Lesion Boundary Detection Using Canny Edge Detection | ✅ Done |

## 🔍 Lab 1 — Transfer Learning & Classifier Comparison

- Transfer learning models compared: AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, DenseNet121, EfficientNet-B0 — evaluated on Accuracy, Precision, Recall, F1-Score, AUC
- Classifiers on deep features: Logistic Regression, Decision Tree, Random Forest, KNN, Linear SVM, RBF-SVM, XGBoost
- Computational efficiency: Parameters, Model Size, FLOPs, Inference Time, Accuracy — compared across all models

📄 Notebook: `Lab1/Computer_Vision_Lab_Task_1.ipynb`

## 🔍 Lab 2 — Effect of Image Filtering on Skin-Lesion Classification

- Dataset: HAM10000 skin-lesion images
- Models used (top-3 from Lab 1): VGG16, EfficientNet-B0, ResNet50
- Filters applied: Average/Mean, Gaussian, Median, Sharpening, Sobel edge — plus unfiltered baseline
- Comparison: 18 (model × filter) runs evaluated on Macro-F1, Balanced Accuracy, AUC, confusion matrices

📄 Notebook: `Lab2/Lab_Task_02.ipynb`

## 🔍 Lab 3 — Edge Detection Techniques and Their Impact on Classification Performance

- Dataset: ISIC Skin Cancer (9 Classes)
- Edge detectors compared: Sobel (Gx/Gy/magnitude), Prewitt, Laplacian, LoG, Canny
- Noise robustness: Gaussian & Salt-and-Pepper noise, with Gaussian/Median filtering as preprocessing
- Canny parameter tuning: threshold and kernel-size sweep, best configuration auto-selected
- Classification comparison: Raw vs. Filtered vs. Edge-only image representations, evaluated with SVM, Random Forest, KNN, and two CNN architectures
- Key finding: edge-only representations *reduced* classification accuracy compared to Raw/Filtered — lesion color and texture matter more than boundary shape for this dataset

📄 Notebook: `Lab3/Lab_3_computer_vision.ipynb`

## 🔍 Lab 4 — Skin Lesion Boundary Detection Using Canny Edge Detection

- Dataset: HAM10000 (5 images selected automatically)
- Pipeline: Original → Grayscale → Gaussian filter (5×5, σ = 1.4) → Canny → Lesion boundary
- Canny threshold settings compared: 50–100, 100–200, 150–250 (50–100 selected as the best of the three)
- Boundary extraction: morphological closing, external contours, largest contour kept as the lesion, with Area and Perimeter calculated
- Final comparison: 8 methods (Original / Average / Gaussian / Median × Sobel / Canny) scored on Noise, Edge Quality (continuity), and Boundary Detection (Dice vs. an Otsu reference mask)
- Key finding: Sobel with smoothing worked best (Average + Sobel scored highest, Dice ≈ 0.69), while Canny with the tested thresholds detected only partial boundaries because HAM10000 lesion borders are soft and low-contrast — lower thresholds, hair removal, and better post-processing are suggested as improvements

📄 Notebook: `Lab4/CV_Lab_4.ipynb`
📝 Answers & report: `Lab4/Lab_Assignment_Answers.md`

## 🛠️ Tools & Libraries

- Python, PyTorch / TorchVision
- scikit-learn, XGBoost
- OpenCV, NumPy, Pandas, Matplotlib, SciPy
- TensorFlow / Keras
- Google Colab

## 👤 Author

Prevesh Maryam
Final Year BS AI Student — COMSATS University Islamabad, Wah Campus
GitHub: [@Maryam-sudo-png](https://github.com/Maryam-sudo-png)
