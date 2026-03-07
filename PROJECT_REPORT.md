# Rice Classification using Machine Learning — Project Report

**Project Title:** Rice Variety Classification System using Image-Based Feature Extraction and ML  
**Domain:** Machine Learning, Computer Vision, Web Application  
**Tech Stack:** Python (Flask, scikit-learn, OpenCV), React (Material-UI), REST API

---

## 1. Introduction

### 1.1 Background

Rice is a staple food for a large population worldwide, and India is one of the largest producers and consumers. Identifying rice varieties accurately is important for quality control, pricing, export compliance, and consumer trust. Manual identification is time-consuming and subjective. This project develops an automated **Rice Classification System** that uses image-based feature extraction and machine learning to classify rice grains into **13 Indian rice varieties**.

### 1.2 Problem Statement

To build an end-to-end system that:
- Accepts rice grain images (upload or camera capture)
- Extracts morphological and visual features (color, texture, size, shape)
- Classifies the rice variety using a trained ML model
- Provides a web dashboard for prediction, history, comparison, marketplace, and admin functions

### 1.3 Objectives

- Design and implement image preprocessing and feature extraction for rice grain images
- Train and evaluate multiple classifiers (Random Forest, SVM, KNN, Gradient Boosting, XGBoost, LightGBM, Ensemble) for 13 Indian rice varieties
- Develop a Flask REST API for prediction, comparison, and dashboard data
- Build a React-based dashboard with Predict, Compare, History, Marketplace, Dataset, Analysis, and Admin pages
- Validate predictions with real-world images and maintain prediction history

---

## 2. System Overview

### 2.1 Architecture

- **Backend:** Flask API server (`api_server.py`) — handles uploads, feature extraction, model prediction, prediction history, listings, admin exports
- **ML Pipeline:** `src/image_preprocessing.py` (feature extraction), `train_all_13_varieties.py` (training), models stored in `models/` (best_model.pkl, scaler.pkl, model_metadata.json)
- **Frontend:** React app in `rice-ml-dashboard/` — Material-UI, React Router, Framer Motion; consumes API at `http://localhost:5001/api`
- **Data Storage:** JSON files in `results/` (prediction_history.json, rice_listings.json); images in `uploads/` and `rice-ml-dashboard/public/prediction_images/`

### 2.2 Rice Varieties (13 Classes)

| # | Variety        | # | Variety          |
|---|----------------|---|------------------|
| 1 | Basmati rice   | 8 | Ponni rice       |
| 2 | Colam rice     | 9 | Jasmine rice     |
| 3 | Indrayani rice |10 | Bamboo rice      |
| 4 | Joha rice      |11 | Mogara rice      |
| 5 | Matta rice     |12 | Brown rice        |
| 6 | Sona Masuri rice | 7 | Kala Jira rice, Ambemohar rice |

### 2.3 Feature Set (25 features)

- **Color:** Mean R, G, B; Mean H, S, V (HSV)
- **Texture:** GLCM (Contrast, Dissimilarity, Homogeneity, Energy); LBP (Mean, Std)
- **Size:** Area, Perimeter
- **Shape:** MajorAxisLength, MinorAxisLength, ConvexArea, Eccentricity, Extent, Roundness, AspectRatio, EquivDiameter, Solidity

---

## 3. Methodology

### 3.1 Image Preprocessing

- Load image (file path or array); convert BGR to RGB; grayscale for texture
- Binary mask via Otsu thresholding; invert if background is darker
- Contour detection to isolate rice grain region for feature extraction

### 3.2 Feature Extraction

- **Color:** Mean RGB and HSV over the masked region
- **Texture:** GLCM on masked grayscale; LBP for texture statistics
- **Size/Shape:** Region properties (area, perimeter, axes, convex area, eccentricity, extent, roundness, aspect ratio, equivalent diameter, solidity)

### 3.3 Model Training

- Data: Images from `data/images/<variety_name>/` and optionally from prediction history
- Pipeline: Extract features → build CSV → train/test split (stratified) → StandardScaler → train multiple classifiers → select best by test accuracy → save best model and scaler
- Models tried: Random Forest, SVM, KNN, Gradient Boosting, LightGBM, XGBoost, Ensemble (VotingClassifier soft)
- Output: `models/best_model.pkl`, `models/scaler.pkl`, `models/model_metadata.json`

### 3.4 Prediction and Validation

- API validates image type and size; rejects non-rice images using heuristic checks (contour area, aspect ratio, color dominance)
- Feature vector is scaled and passed to the loaded model; top-3 predictions with confidence returned
- Prediction stored in history with timestamp, variety, confidence, image path, and extracted features

---

## 4. Implementation

### 4.1 API Endpoints (Summary)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/api/health` | GET | Health check, model/scaler status, varieties |
| `/api/predict` | POST | Upload image → prediction + top-3 + features |
| `/api/extract-features` | POST | Extract features only (no prediction) |
| `/api/compare` | POST | Compare two images (predictions + features) |
| `/api/rice-varieties` | GET | List of 13 varieties |
| `/api/dashboard-data` | GET | Project/dataset/model/prediction stats for dashboard |
| `/api/prediction-stats` | GET | Aggregate prediction statistics |
| `/api/prediction-history` | GET | Full prediction history |
| `/api/images/<filename>` | GET | Serve prediction image |
| `/api/clear-history` | POST | Clear prediction history |
| `/api/listings` | GET/POST | Rice marketplace listings |
| `/api/listings/<id>/purchase` | POST | Record purchase |
| `/api/admin/stats` | GET | Admin statistics |
| `/api/admin/export/purchases` | GET | Export purchases CSV |
| `/api/admin/export/listings` | GET | Export listings CSV |

### 4.2 Dashboard Pages

- **Predict:** Upload image or capture via camera; display prediction, confidence, top-3, and extracted features
- **Compare:** Upload two images and compare predictions and features side by side
- **History:** List past predictions with thumbnails, variety, confidence, timestamp; optional clear
- **Marketplace:** List/sell rice with contact/email; purchase list
- **Dataset:** Show dataset overview (class distribution, samples, features) from dashboard data
- **Analysis:** Model performance, metrics, feature importance (from dashboard data)
- **Admin:** Stats, export purchases/listings

### 4.3 Key Files

| Path | Purpose |
|------|---------|
| `api_server.py` | Flask app, routes, prediction logic, history, listings, admin |
| `src/image_preprocessing.py` | RiceImageFeatureExtractor, INDIAN_RICE_VARIETIES |
| `train_all_13_varieties.py` | Data collection, feature extraction, training, evaluation, save model |
| `rice-ml-dashboard/src/App.js` | Router, theme, dashboard layout, menu |
| `rice-ml-dashboard/src/pages/PredictPage.js` | Upload/camera, preview, predict, display results |
| `models/best_model.pkl` | Trained classifier (joblib) |
| `models/scaler.pkl` | StandardScaler (joblib) |
| `results/prediction_history.json` | All predictions with metadata |

---

## 5. Project Screenshots

*Add actual screenshots in the same order below. Suggested locations: create a `docs/screenshots/` folder and place images there, then reference them in this report.*

### 5.1 Dashboard – Home / Navigation

**[INSERT SCREENSHOT: Dashboard layout with sidebar navigation (Predict, Compare, History, Marketplace, Dataset, Analysis, Admin) and app bar title.]**  
*Suggested filename: `docs/screenshots/01-dashboard-nav.png`*

### 5.2 Predict Page – Upload

**[INSERT SCREENSHOT: Predict page showing upload area (drag-and-drop or file picker) and “Choose Image” / “Capture from Camera” buttons.]**  
*Suggested filename: `docs/screenshots/02-predict-upload.png`*

### 5.3 Predict Page – After Prediction

**[INSERT SCREENSHOT: Predict page after a successful prediction: uploaded/captured image, predicted variety name, confidence %, top-3 list, and extracted features table.]**  
*Suggested filename: `docs/screenshots/03-predict-result.png`*

### 5.4 Predict Page – Camera Capture

**[INSERT SCREENSHOT: Camera capture dialog with live video and capture button.]**  
*Suggested filename: `docs/screenshots/04-predict-camera.png`*

### 5.5 Compare Page

**[INSERT SCREENSHOT: Compare page with two images and their prediction results side by side.]**  
*Suggested filename: `docs/screenshots/05-compare.png`*

### 5.6 History Page

**[INSERT SCREENSHOT: History page showing list/cards of past predictions with thumbnails, variety, confidence, and date.]**  
*Suggested filename: `docs/screenshots/06-history.png`*

### 5.7 Marketplace Page

**[INSERT SCREENSHOT: Marketplace page with rice listings and sell/purchase options.]**  
*Suggested filename: `docs/screenshots/07-marketplace.png`*

### 5.8 Dataset Page

**[INSERT SCREENSHOT: Dataset page showing class distribution and dataset statistics.]**  
*Suggested filename: `docs/screenshots/08-dataset.png`*

### 5.9 Analysis Page

**[INSERT SCREENSHOT: Analysis page showing model performance metrics and charts.]**  
*Suggested filename: `docs/screenshots/09-analysis.png`*

### 5.10 Admin Page

**[INSERT SCREENSHOT: Admin page with stats and export buttons.]**  
*Suggested filename: `docs/screenshots/10-admin.png`*

### 5.11 Sample Prediction Images (from project)

Prediction images saved by the system are stored in:
- `rice-ml-dashboard/public/prediction_images/` (for frontend display)
- `uploads/` (backend copy)

Example naming: `prediction_YYYYMMDD_HHMMSS_<original_filename>.jpg`  
*You may add 1–2 sample screenshots from these folders showing real prediction thumbnails in the History page.*

---

## 6. Results and Discussion

- **Model:** Best model (e.g., Random Forest/Ensemble) is saved after training; accuracy, precision, recall, F1, and ROC-AUC are in `models/model_metadata.json`.
- **Prediction history:** Stored in `results/prediction_history.json`; variety counts and recent predictions are exposed via API for the dashboard.
- **Validation:** Non-rice images are rejected with a clear error message (e.g., “This doesn’t look like a rice grain image…”).
- **Limitations:** Accuracy depends on training data per variety; single-grain or clear background images work best.

---

## 7. Conclusion

The project delivers a complete **Rice Classification System** with:
- Robust feature extraction from rice grain images (color, texture, size, shape)
- Multi-class ML pipeline for 13 Indian rice varieties and a saved best model
- REST API for prediction, comparison, history, marketplace, and admin
- React dashboard with Predict (upload + camera), Compare, History, Marketplace, Dataset, Analysis, and Admin

Future work can include: more varieties, deep learning (CNN) options, mobile app, and integration with weighing/labeling devices.

---

## 8. References

- scikit-learn: https://scikit-learn.org/
- OpenCV: https://opencv.org/
- Flask: https://flask.palletsprojects.com/
- React: https://react.dev/
- Material-UI: https://mui.com/

---

## 9. Appendix

### A. How to Run the Project

1. **Backend:**  
   `cd /path/to/Rice-Classification-ML`  
   `python -m venv venv` (if not already)  
   `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)  
   `pip install -r requirements.txt`  
   `python api_server.py` (default port 5001)

2. **Frontend:**  
   `cd rice-ml-dashboard`  
   `npm install`  
   `npm start` (default port 3000)

3. **Training (optional):**  
   Place images in `data/images/<variety_name>/` then run:  
   `python train_all_13_varieties.py`

### B. Dependencies (Key)

- Python: pandas, numpy, scikit-learn, opencv-python, scikit-image, joblib, flask, flask-cors, xgboost, lightgbm, etc. (see `requirements.txt`)
- Node: React, React Router, Material-UI, Framer Motion (see `rice-ml-dashboard/package.json`)

---

*End of Project Report*
