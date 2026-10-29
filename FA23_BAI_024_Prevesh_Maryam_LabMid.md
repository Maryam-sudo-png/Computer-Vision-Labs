# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images

**Computer Vision Lab** | Reg No: FA23-BAI-024 | Name: Prevesh Maryam

| Setting | Rule | My value |
|---|---|---|
| Random seed | last two digits (024) | **24** |
| HOG cell size | last digit 4 = even | **8 x 8** |
| SIFT grid | last digit 4 (0 to 4) | **4 x 4 grid (16 keypoints)** |

---

## Cell 1: Personal Settings

**Output:**
```
Seed=24, HOG cell=(8, 8), SIFT grid=4x4 (16 keypoints)
```


# Q1. Data Setup (3 marks)

**Output:**
```
Train | X shape: (400, 128, 128) | y shape: (400,) | counts: {'eosinophil': 100, 'monocyte': 100, 'neutrophil': 100, 'platelet': 100}
Val   | X shape: (120, 128, 128) | y shape: (120,) | counts: {'eosinophil': 30, 'monocyte': 30, 'neutrophil': 30, 'platelet': 30}
Test  | X shape: (120, 128, 128) | y shape: (120,) | counts: {'eosinophil': 30, 'monocyte': 30, 'neutrophil': 30, 'platelet': 30}
```

![q1_examples](images/q1_examples.png)


# Q2. HOG + SVM (6 marks)


### Q2(a) HOG feature length (by hand)
- Image = 128 x 128, cell = 8 x 8, so cells per side = 128 / 8 = **16**
- Block = 2 x 2 cells, stride = 1 cell, so blocks per side = 16 - 1 = **15**, total blocks = 15 x 15 = **225**
- Values per block = 2 x 2 cells x 9 orientations = **36**
- Total HOG length = 225 x 36 = **8100**

**Output:**
```
HOG feature length: 8100 | train: (400, 8100) | val: (120, 8100)
Hand calculation check: 15*15*2*2*9 = 8100
```


### Q2(b) HOG picture (one image per class)

![q2_hog_pictures](images/q2_hog_pictures.png)


### Q2(c) Train SVM (C chosen on validation)

**Output:**
```
--- HOG + SVM: C selection on validation ---
   C  val_acc
 0.1 0.725000
 1.0 0.716667
10.0 0.708333
HOG + SVM: best C=0.1 | val accuracy=0.7250 | val macro-F1=0.7267
```

![q2_confusion_hog](images/q2_confusion_hog.png)

**Output:**
```
[[20  5  5  0]
 [10 17  3  0]
 [ 5  4 21  0]
 [ 0  1  0 29]]
```


**Q2(c) Results (HOG + SVM, validation set):**

| C | Val accuracy |
|---|---|
| 0.1 | 0.7250 |
| 1 | 0.7167 |
| 10 | 0.7083 |

- Best **C = 0.1**, validation accuracy = **0.7250 (72.5%)**, macro-F1 = **0.7267**
- Confusion matrix (rows = true, columns = predicted; order: eosinophil, monocyte, neutrophil, platelet):

| True \ Pred | eosinophil | monocyte | neutrophil | platelet |
|---|---|---|---|---|
| eosinophil | 20 | 5 | 5 | 0 |
| monocyte | 10 | 17 | 3 | 0 |
| neutrophil | 5 | 4 | 21 | 0 |
| platelet | 0 | 1 | 0 | 29 |

Platelets are almost perfectly classified (29/30), while the three white-cell classes are confused with each other.


# Q3. SIFT + SVM (7 marks)


### Q3(a) Normal SIFT detection on training images

**Output:**
```
Average keypoints/image: 73.33
Min: 10 | Max: 294
Images with zero keypoints: 0 of 400
```

![q3_sift_keypoints](images/q3_sift_keypoints.png)

**Why is a different number of keypoints a problem for an SVM?**

The output shows the number of keypoints varies a lot between images: average **73.33**, minimum **10**, maximum **294**, and **0 images** with zero keypoints. Every keypoint gives a 128-d descriptor, so one image would produce a vector of length 10 x 128 and another of 294 x 128. An SVM needs every sample to be a vector of the same fixed length, so these variable-length outputs cannot be fed to it directly.

### Q3(b) Fixed grid SIFT
Keypoints are placed on a regular 4 x 4 grid (cell centres at 16, 48, 80, 112) over the 128 x 128 image, keypoint size = 32.

**Feature length by hand:** 4 x 4 = 16 grid points, each descriptor = 128 values, so 16 x 128 = **2048** (matches the printed output).

**Output:**
```
Grid coordinates: [16.0, 48.0, 80.0, 112.0] | number of keypoints: 16
SIFT feature length: 2048 | train: (400, 2048) | val: (120, 2048)
Hand calculation check: 16*128 = 2048
```

![q3_grid_keypoints](images/q3_grid_keypoints.png)


### Q3(c) Train SVM (C chosen on validation)

**Output:**
```
--- SIFT + SVM: C selection on validation ---
   C  val_acc
 0.1 0.716667
 1.0 0.716667
10.0 0.716667
SIFT + SVM: best C=0.1 | val accuracy=0.7167 | val macro-F1=0.7249
```

![q3_confusion_sift](images/q3_confusion_sift.png)

**Output:**
```
[[19  8  3  0]
 [11 19  0  0]
 [ 9  0 20  1]
 [ 1  0  1 28]]
```


**Q3(c) Results (SIFT + SVM, validation set):**

| C | Val accuracy |
|---|---|
| 0.1 | 0.7167 |
| 1 | 0.7167 |
| 10 | 0.7167 |

- All three C values gave the same validation accuracy, so the first (smallest) one, **C = 0.1**, was kept: validation accuracy = **0.7167 (71.7%)**, macro-F1 = **0.7249**
- Confusion matrix (rows = true, columns = predicted):

| True \ Pred | eosinophil | monocyte | neutrophil | platelet |
|---|---|---|---|---|
| eosinophil | 19 | 8 | 3 | 0 |
| monocyte | 11 | 19 | 0 | 0 |
| neutrophil | 9 | 0 | 20 | 1 |
| platelet | 1 | 0 | 1 | 28 |

*Note:* sklearn printed a `ConvergenceWarning` (liblinear did not fully converge on the unscaled 2048-d SIFT features within 20000 iterations). The accuracy values are still valid outputs of the trained models, but this is why all three C values behave the same.


# Q4. Robustness (6 marks)


### Q4(a) Prediction (written before running the test evaluation)
- **Rotation:** I think **HOG** will lose more accuracy. HOG uses a fixed grid of cell-wise gradient histograms, so rotating by 20 degrees shifts the edge orientations into different bins and cells. The grid SIFT descriptor is also not rotation-invariant here (orientation is computed from the grid keypoint's angle, which is fixed at 0 and not re-estimated), but its 32 px support window with Gaussian weighting is smoother, so I expect it to be a bit more tolerant.
- **Noise:** I think **SIFT** will lose more accuracy. SIFT builds histograms from raw pixel gradients over only 16 locations, and noise adds random gradients that directly change those histograms. HOG averages over many cells and block-normalises, which damps noise.


### Q4(b) Evaluation on test set (models frozen, trained on train only)

![q4_clean_rotated_noisy](images/q4_clean_rotated_noisy.png)

**Output:**
```
            Clean accuracy  Rotated accuracy  Noisy accuracy
HOG + SVM           0.6667            0.6333          0.5417
SIFT + SVM          0.7583            0.3000          0.7583

            Drop (rotation)  Drop (noise)
HOG + SVM            0.0333         0.125
SIFT + SVM           0.4583         0.000
```


### Q4(c) Was my prediction correct?

**No, both parts of my prediction were wrong.**

**Final table (test set, 120 images, models frozen):**

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---|---|---|
| HOG + SVM | 0.6667 | 0.6333 | 0.5417 |
| SIFT + SVM | 0.7583 | 0.3000 | 0.7583 |

**Explanation (3 sentences):**
1. Under rotation, SIFT + SVM lost far more accuracy (0.7583 to 0.3000, a drop of 0.4583) than HOG + SVM (0.6667 to 0.6333, a drop of 0.0333), so my rotation prediction (HOG) was wrong; this is probably because the fixed-grid SIFT descriptor stores raw gradient orientations at 16 fixed positions, so a 20-degree turn changes its histograms a lot, while HOG's coarse 8x8 cell histograms and block normalisation shifted less.
2. Under noise, HOG + SVM lost accuracy (0.6667 to 0.5417, a drop of 0.1250) while SIFT + SVM stayed at 0.7583 (drop 0.0000), so my noise prediction (SIFT) was also wrong; SIFT's descriptor smooths the 32 px window with Gaussian weighting, which averages out the random noise, whereas HOG takes gradients of fine 8x8 cells directly from the noisy pixels.
3. Overall, SIFT + SVM is the better model on clean and noisy images but fails badly under rotation, while HOG + SVM is less accurate but more stable under rotation, so neither model is reliable for everything and the best choice depends on whether rotation or noise is the bigger problem.

**Output:**
```
Rotation: SIFT + SVM lost more accuracy (HOG drop=0.0333, SIFT drop=0.4583).
Noise: HOG + SVM lost more accuracy (HOG drop=0.1250, SIFT drop=0.0000).
My prediction -> rotation: HOG, noise: SIFT
Rotation prediction correct? False
Noise prediction correct?    False
```


# Q5. Short Answers and AI Log (3 marks)


### Q5(a) Why does HOG use block normalisation?
Gradient magnitudes change with lighting and contrast, so raw cell histograms for the same cell structure can differ a lot between images. Block normalisation divides each block's histogram by its own norm (L2-Hys here), which makes the features much less sensitive to illumination and contrast changes. Overlapping blocks also let each cell be normalised in several contexts, which improves stability.

### Q5(b) Why is it wrong to choose C using the test set?
The test set must stay unseen so that it gives an honest estimate of performance on new data. If C is picked to maximise test accuracy, information from the test set leaks into the model choice, so the reported accuracy becomes optimistically biased. C should be chosen on the validation set, and the test set used only once at the end (Q4).

### Q5(c) Most confused classes

**Output:**
```
HOG + SVM (validation): most confused = monocyte vs eosinophil | monocyte->eosinophil: 10, eosinophil->monocyte: 5
SIFT + SVM (validation): most confused = monocyte vs eosinophil | monocyte->eosinophil: 11, eosinophil->monocyte: 8
HOG + SVM (clean test): most confused = neutrophil vs eosinophil | neutrophil->eosinophil: 12, eosinophil->neutrophil: 5
SIFT + SVM (clean test): most confused = eosinophil vs monocyte | eosinophil->monocyte: 9, monocyte->eosinophil: 6
```

**Answer:**

The printed outputs show **eosinophil** is involved in the most confusion in every case, and it is mostly mixed up with **monocyte**:

| Model / split | Most confused pair | Counts |
|---|---|---|
| HOG + SVM (validation) | monocyte vs eosinophil | monocyte to eosinophil: 10, eosinophil to monocyte: 5 |
| SIFT + SVM (validation) | monocyte vs eosinophil | 11 and 8 |
| HOG + SVM (clean test) | neutrophil vs eosinophil | neutrophil to eosinophil: 12, eosinophil to neutrophil: 5 |
| SIFT + SVM (clean test) | eosinophil vs monocyte | eosinophil to monocyte: 9, monocyte to eosinophil: 6 |

So the two classes my models confused most overall are **eosinophil and monocyte** (eosinophil and neutrophil is the second pair, seen once for HOG on the test set).

**Why grayscale makes this harder:** these are all white blood cells with a similar round shape and a nucleus. In colour images, an eosinophil is mainly recognised by its bright red-orange granules, and a monocyte by its larger, pale bluish-gray cytoplasm. Converting to grayscale merges all colours into one brightness channel, so these colour cues are lost and the models have to separate the cells using only brightness, texture and shape, which overlap between these classes. Platelets are small and look very different, which is why they are almost always classified correctly (29/30 and 28/30 on validation).

### Q5(d) AI Log

| Prompt (one line) | What I changed | How I checked the result was correct |
|---|---|---|
| "Solve the HOG vs SIFT lab mid for reg no 024 (seed 24, HOG 8x8, SIFT 4x4 grid)" | Used my settings from the registration-number rule; ran the generated notebook myself and replaced the written prediction/answers with explanations based on my own printed numbers | Printed shapes (400/120/120 with 100/30/30 per class), HOG length 8100 and SIFT length 2048 matched the hand calculations, looked at the HOG pictures, grid keypoints and rotated/noisy sample images, and confirmed accuracy numbers from the confusion matrices (e.g. 87/120 = 0.7250) |

---
**Rules check:** C was chosen on the validation set only; the test set was used only in Q4; every number above comes from the printed outputs in this notebook.
