# Computer Vision Labs

Lab tasks for the **Computer Vision** course — BS Artificial Intelligence, COMSATS University Islamabad, Wah Campus.

Each lab lives in its own folder and gets added here as the semester progresses.

## 📂 Structure

| Lab | Topic | Status |
|---|---|---|
| [Lab1](./Lab1) | Transfer Learning Models, Classifier & Computational Efficiency Comparison | ✅ Done |
| [Lab2](./Lab2) | Effect of Image Filtering on Skin-Lesion Classification | ✅ Done |

## 🔍 Lab 1 — Transfer Learning & Classifier Comparison

- **Transfer learning models compared:** AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, DenseNet121, EfficientNet-B0 — evaluated on Accuracy, Precision, Recall, F1-Score, AUC
- **Classifiers on deep features:** Logistic Regression, Decision Tree, Random Forest, KNN, Linear SVM, RBF-SVM, XGBoost
- **Computational efficiency:** Parameters, Model Size, FLOPs, Inference Time, Accuracy — compared across all models

📄 Notebook: [`Lab1/Computer_Vision_Lab_Task_1.ipynb`](./Lab1/Computer_Vision_Lab_Task_1.ipynb)

## 🔍 Lab 2 — Effect of Image Filtering on Skin-Lesion Classification

- **Dataset:** HAM10000 skin-lesion images
- **Models used (top-3 from Lab 1):** VGG16, EfficientNet-B0, ResNet50
- **Filters applied:** Average/Mean, Gaussian, Median, Sharpening, Sobel edge — plus unfiltered baseline
- **Comparison:** 18 (model × filter) runs evaluated on Macro-F1, Balanced Accuracy, AUC, confusion matrices

📄 Notebook: [`Lab2/Lab_Task_02.ipynb`](./Lab2/Lab_Task_02.ipynb)

## 🛠️ Tools & Libraries

- Python, PyTorch / TorchVision
- scikit-learn, XGBoost
- Google Colab

## 👤 Author

**Prevesh Maryam**
Final Year BS AI Student — COMSATS University Islamabad, Wah Campus
GitHub: [@Maryam-sudo-png](https://github.com/Maryam-sudo-png)
