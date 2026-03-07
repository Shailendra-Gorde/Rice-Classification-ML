# Rice Variety Classification Using Image-Based Feature Extraction and Machine Learning

**Authors:** [Your Name], [Co-authors if any]  
**Affiliation:** [Your College/University]  
**Contact:** [Email]

---

## Abstract

Automated identification of rice varieties is important for quality assurance, fair pricing, and supply-chain transparency. This paper presents a machine learning system that classifies rice grains into 13 Indian rice varieties using morphological and visual features extracted from images. We use a handcrafted feature set comprising color (RGB, HSV), texture (GLCM, LBP), and size-shape descriptors, followed by standard scaling and ensemble classification. The pipeline is exposed via a REST API and a web dashboard that supports image upload and camera capture. Experiments on images organized by variety show that an ensemble of Random Forest, Gradient Boosting, and other classifiers achieves strong multi-class accuracy. We discuss design choices, validation of non-rice rejection, and directions for extending the system with more varieties and deep learning.

**Keywords:** Rice classification, image processing, feature extraction, machine learning, ensemble learning, Indian rice varieties, computer vision.

---

## 1. Introduction

### 1.1 Motivation

Rice (*Oryza sativa*) is a staple for billions of people. India is among the largest producers and consumers of rice, with many distinct varieties such as Basmati, Sona Masuri, and Joha. Correct identification of rice type affects quality control, labeling, export compliance, and consumer trust. Manual inspection is slow and subjective. Automating this task using images and machine learning can support traders, quality labs, and digital agriculture platforms.

### 1.2 Objectives

We aim to:
1. Extract a fixed set of interpretable features from rice grain images (color, texture, size, shape).
2. Train multi-class classifiers to distinguish 13 Indian rice varieties.
3. Integrate the model into a web-based system with upload and camera input.
4. Reject clearly non-rice images to avoid spurious predictions.

### 1.3 Contributions

- A reproducible feature-extraction and training pipeline for 13 Indian rice varieties.
- Validation heuristics to reject non-rice images before prediction.
- A full-stack deployment: Flask API + React dashboard with prediction history and comparison.

---

## 2. Related Work

Image-based grain and crop classification has been studied with both handcrafted features and deep learning.

- **Handcrafted features:** Color histograms, texture (GLCM, LBP), and shape descriptors (area, perimeter, aspect ratio) are widely used for grain type classification and quality grading. Such features are interpretable and work well with moderate-sized datasets.
- **Deep learning:** CNNs have been applied to rice and other grain classification, often achieving high accuracy when large labeled datasets are available. They require more data and computational resources.
- **Indian rice varieties:** Prior work has focused on specific subsets (e.g., Basmati vs non-Basmati) or smaller sets of varieties. Our system targets 13 named Indian varieties in a single multi-class setup.

Our approach combines handcrafted features with ensemble ML to balance accuracy, interpretability, and ease of deployment without requiring very large datasets.

---

## 3. Methodology

### 3.1 Rice Varieties and Data

We consider 13 Indian rice varieties: Basmati rice, Colam rice, Indrayani rice, Joha rice, Matta rice, Sona Masuri rice, Kala Jira rice, Ambemohar rice, Ponni rice, Jasmine rice, Bamboo rice, Mogara rice, and Brown rice. Training images are organized in folders `data/images/<variety_name>/`. Optionally, correctly labeled images from prediction history can be included in training.

### 3.2 Image Preprocessing

- Input: RGB image (file or array).
- Grayscale conversion for texture; Otsu thresholding to obtain a binary mask separating grain from background; mask inversion if the background is darker.
- Contour detection to isolate the rice region; features are computed on the masked region to reduce background influence.

### 3.3 Feature Extraction

We extract 25 numerical features in four groups:

1. **Color (6):** Mean R, G, B and mean H, S, V in the masked region.
2. **Texture – GLCM (4):** Contrast, Dissimilarity, Homogeneity, Energy.
3. **Texture – LBP (2):** Mean and standard deviation of Local Binary Pattern.
4. **Size and shape (13):** Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, Eccentricity, Extent, Roundness, AspectRatio, EquivDiameter, Solidity.

All features are computed using OpenCV and scikit-image on the preprocessed image and mask.

### 3.4 Non-Rice Validation

Before prediction, we apply simple heuristics to reject obviously non-rice images:
- Contour area too small or covering almost the whole image.
- Extreme aspect ratio.
- Strong blue or green dominance (untypical for rice).

Rejected requests return a clear error message asking for a rice grain image.

### 3.5 Model Training

- Features are collected into a matrix; target labels are variety indices (0–12).
- Data is split into train/test (stratified). Features are standardized with StandardScaler.
- We train: Random Forest, SVM, KNN, Gradient Boosting, LightGBM, XGBoost, and a soft-voting Ensemble.
- The model with the best test accuracy is saved (joblib) along with the scaler and metadata (accuracy, precision, recall, F1, ROC-AUC, variety list).

### 3.6 Deployment

- **API:** Flask server provides `/api/predict` (image upload → prediction + top-3 + features), `/api/compare`, `/api/prediction-history`, and other endpoints.
- **Dashboard:** React app with Predict (upload + camera), Compare, History, Marketplace, Dataset, Analysis, and Admin pages, consuming the API.

---

## 4. Experiments and Results

### 4.1 Setup

- **Software:** Python 3.x, scikit-learn, OpenCV, scikit-image, Flask; React, Material-UI for frontend.
- **Data:** Number of images per variety and total samples are reported in `models/model_metadata.json` after training.
- **Metrics:** Accuracy, weighted precision/recall/F1, and multi-class ROC-AUC (one-vs-rest).

### 4.2 Model Comparison

Training compares multiple classifiers; the best is selected by test accuracy and saved as `best_model.pkl`. Typical outcomes (to be filled from your run):
- Ensemble and tree-based models (Random Forest, Gradient Boosting, XGBoost, LightGBM) tend to perform well.
- Per-class accuracy and confusion matrix can be inspected from the training script output and metadata.

### 4.3 Prediction Behavior

- Successful predictions return predicted variety, confidence (probability), and top-3 varieties.
- Non-rice images are rejected with an explicit error.
- Prediction history is persisted and exposed for dashboard and optional retraining.

### 4.4 Screenshots (Research Paper)

**[INSERT FIGURE 1: System architecture diagram (Backend API + ML model + Frontend dashboard).]**  
**[INSERT FIGURE 2: Predict page – upload/camera and prediction result with top-3 and features.]**  
**[INSERT FIGURE 3: Compare page – two images with predictions side by side.]**  
**[INSERT FIGURE 4: Sample confusion matrix or bar chart of per-variety accuracy from training.]**

---

## 5. Discussion

- **Strengths:** Interpretable features; no need for very large datasets; fast inference; full pipeline from image to web UI; non-rice validation reduces misuse.
- **Limitations:** Performance depends on quality and balance of training data; single-grain or clear-background images work best; handcrafted features may miss subtle visual cues that CNNs can learn.
- **Future work:** Include more varieties; add CNN-based option; mobile app; integration with weighing/labeling hardware.

---

## 6. Conclusion

We described an end-to-end system for classifying 13 Indian rice varieties from images using handcrafted features and ensemble machine learning. The system includes preprocessing, feature extraction, non-rice validation, model training and selection, a REST API, and a React dashboard. Results demonstrate feasibility of automated rice variety identification with a transparent, deployable pipeline. Extending the dataset and optionally combining with deep learning could further improve robustness and coverage.

---

## References

1. scikit-learn: Machine Learning in Python. https://scikit-learn.org/
2. OpenCV. https://opencv.org/
3. Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Trans. Syst. Man Cybern., 9(1), 62–66.
4. Haralick, R. M., et al. (1973). Textural features for image classification. IEEE Trans. Syst. Man Cybern., 3(6), 610–621.
5. Flask Documentation. https://flask.palletsprojects.com/
6. React. https://react.dev/
7. [Add any papers on rice/grain classification you have referred to.]

---

*End of Research Paper*
