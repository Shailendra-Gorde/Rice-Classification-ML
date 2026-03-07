# Blackbook — Rice Classification using Machine Learning

**Project Title:** Rice Variety Classification System using Image-Based Feature Extraction and Machine Learning  
**Academic Year:** [e.g., 2025–2026]  
**Course:** [e.g., B.E. / B.Tech. in Computer Engineering / IT]  
**University:** [Your University Name]

---

## Certificate

This is to certify that the project entitled **“Rice Variety Classification System using Image-Based Feature Extraction and Machine Learning”** submitted by ** [Name of Student(s)] ** in partial fulfilment of the requirements for the award of ** [Degree Name] ** in ** [Branch] ** is a record of bonafide work carried out under my/our supervision. The matter embodied in this project has not been submitted earlier for the award of any other degree/diploma.

**Place:** _______________  
**Date:** _______________

_________________________  
**Signature of Guide**

_________________________  
**Signature of HOD**

_________________________  
**Signature of External Examiner**

---

## Acknowledgement

We would like to express our sincere gratitude to our project guide ** [Guide Name] ** and the department faculty for their guidance and support throughout this project. We thank our institution and the laboratory staff for providing the necessary resources. We also thank our colleagues and family for their encouragement. Finally, we thank the open-source communities of Python, scikit-learn, OpenCV, Flask, and React for the tools that made this system possible.

---

## Abstract

This project presents an automated **Rice Variety Classification System** that identifies 13 Indian rice varieties from grain images using image processing and machine learning. The system extracts color, texture, size, and shape features from rice images, trains ensemble classifiers, and serves predictions through a Flask REST API. A React-based web dashboard allows users to upload images or capture via camera, view predictions with confidence and top-3 results, compare two images, and browse prediction history. Additional modules include a rice marketplace (listings and purchases) and an admin panel for statistics and exports. The project demonstrates a complete pipeline from image input to classification and deployment, with validation to reject non-rice images. Technologies used include Python, OpenCV, scikit-learn, Flask, React, and Material-UI.

**Keywords:** Rice classification, image processing, feature extraction, machine learning, Flask API, React dashboard, Indian rice varieties.

---

## Table of Contents

1. **Introduction**
   - 1.1 Background and motivation  
   - 1.2 Problem statement  
   - 1.3 Objectives  
   - 1.4 Scope  
   - 1.5 Organization of report  

2. **System Analysis**
   - 2.1 Existing system and limitations  
   - 2.2 Proposed system  
   - 2.3 Feasibility study  
   - 2.4 Hardware and software requirements  

3. **System Design**
   - 3.1 Architecture overview  
   - 3.2 Module design  
   - 3.3 Data flow  
   - 3.4 Database / data storage design  

4. **Implementation**
   - 4.1 Technology stack  
   - 4.2 Feature extraction implementation  
   - 4.3 Model training pipeline  
   - 4.4 API implementation  
   - 4.5 Frontend implementation  

5. **Testing**
   - 5.1 Test strategy  
   - 5.2 Test cases and results  
   - 5.3 Screenshots of working system  

6. **Conclusion and Future Work**
   - 6.1 Conclusion  
   - 6.2 Limitations  
   - 6.3 Future enhancements  

7. **References / Bibliography**

8. **Appendix**
   - A. Sample code snippets  
   - B. Screenshots (all project screens)  
   - C. API endpoint list  
   - D. User manual (brief)  

---

# Chapter 1 — Introduction

## 1.1 Background and Motivation

Rice is a staple food for a large part of the world’s population. India is one of the largest producers and consumers of rice, with many distinct varieties such as Basmati, Sona Masuri, Joha, and Indrayani. Correct identification of rice type is important for quality control, fair pricing, export standards, and consumer trust. Manual identification is time-consuming and can be subjective. This project aims to automate rice variety identification using grain images and machine learning, making it useful for traders, quality labs, and digital agriculture.

## 1.2 Problem Statement

To design and implement a system that:
- Accepts rice grain images (file upload or camera capture).
- Extracts meaningful visual and morphological features.
- Classifies the rice into one of 13 Indian rice varieties using a trained ML model.
- Provides a web interface for prediction, comparison, history, marketplace, and administration.
- Rejects clearly non-rice images to avoid wrong predictions.

## 1.3 Objectives

- To implement image preprocessing and feature extraction for rice grain images (color, texture, size, shape).
- To train and evaluate multiple ML models and select the best for 13 Indian rice varieties.
- To develop a REST API for prediction, comparison, and dashboard data.
- To build a React-based dashboard with Predict, Compare, History, Marketplace, Dataset, Analysis, and Admin pages.
- To store prediction history and support optional retraining from verified predictions.

## 1.4 Scope

- **In scope:** 13 fixed Indian rice varieties; image-based classification; web dashboard and API; prediction history; comparison; marketplace and admin.
- **Out of scope:** Real-time video classification; integration with weighing hardware; mobile native app (web is responsive).

## 1.5 Organization of Report

Chapter 2 analyses the existing vs proposed system and requirements. Chapter 3 describes system design and architecture. Chapter 4 details implementation. Chapter 5 covers testing and screenshots. Chapter 6 concludes and outlines future work. References and appendix (screenshots, API list, user manual) follow.

---

# Chapter 2 — System Analysis

## 2.1 Existing System and Limitations

- **Manual identification:** Experts visually inspect rice samples. Limitations: slow, subjective, not scalable.
- **Generic grain analyzers:** Some tools exist for grain quality (moisture, size) but not always for fine-grained variety classification.
- **Research prototypes:** Many papers use small datasets or limited varieties; few provide a full deployable system with API and UI.

## 2.2 Proposed System

- End-to-end pipeline: Image → Preprocessing → Feature extraction → Scaling → Model → Prediction.
- Validation step to reject non-rice images.
- REST API for all operations (predict, compare, history, listings, admin).
- Web dashboard for end users and admins.
- Persisted prediction history and optional use in retraining.

## 2.3 Feasibility Study

- **Technical:** Python and JavaScript ecosystems provide mature libraries (OpenCV, scikit-learn, Flask, React). Feasible.
- **Operational:** Standard PC and server; no special hardware. Feasible.
- **Economic:** Open-source stack; minimal cost. Feasible.

## 2.4 Hardware and Software Requirements

**Hardware:**  
- Processor: Intel Core i5 or equivalent  
- RAM: 8 GB or more  
- Storage: 500 MB for project + models + images  
- Camera (optional): for capture from dashboard  

**Software:**  
- OS: Windows / Linux / macOS  
- Python 3.8+  
- Node.js 16+ (for React app)  
- Browser: Chrome, Firefox, Safari (for dashboard)  
- Dependencies: See `requirements.txt` and `rice-ml-dashboard/package.json`

---

# Chapter 3 — System Design

## 3.1 Architecture Overview

- **Backend:** Flask application (`api_server.py`) — serves REST endpoints, loads model and scaler, uses `RiceImageFeatureExtractor` from `src/image_preprocessing.py`.
- **ML pipeline:** Feature extraction from images → CSV/array → train/test split → StandardScaler → train multiple classifiers → save best model and scaler to `models/`.
- **Frontend:** React SPA in `rice-ml-dashboard/` — calls API at `http://localhost:5001/api`, displays Predict, Compare, History, Marketplace, Dataset, Analysis, Admin.
- **Data:** JSON files in `results/` (prediction_history.json, rice_listings.json); images in `uploads/` and `rice-ml-dashboard/public/prediction_images/`.

## 3.2 Module Design

| Module | Responsibility |
|--------|----------------|
| Image preprocessing | Load image, grayscale, Otsu mask, contour |
| Feature extraction | Color (RGB, HSV), texture (GLCM, LBP), size/shape (region props) |
| Validation | Reject non-rice (area, aspect ratio, color checks) |
| Model service | Load model/scaler, scale features, predict, top-3 |
| API routes | /api/predict, /api/compare, /api/prediction-history, etc. |
| History & listings | Load/save JSON for predictions and marketplace |
| Frontend pages | Predict, Compare, History, Marketplace, Dataset, Analysis, Admin |

## 3.3 Data Flow

1. User uploads/captures image in dashboard → frontend sends POST to `/api/predict`.
2. API saves image, converts to array, extracts features, validates (rice vs non-rice).
3. If valid: scale features → model.predict/proba → return prediction + top-3 + features; append to prediction history and save.
4. Dashboard displays result; history and other pages fetch data from respective API endpoints.

## 3.4 Data Storage Design

- **prediction_history.json:** total_predictions, variety_counts, list of predictions (id, timestamp, variety, confidence, image_filename, image_path, features, top_3_predictions).
- **rice_listings.json:** listings array (id, variety, price, contact, email, etc.) and purchase records as needed.
- **models/:** best_model.pkl, scaler.pkl, model_metadata.json (varieties, accuracy, sample counts, etc.).

---

# Chapter 4 — Implementation

## 4.1 Technology Stack

- **Backend:** Python 3, Flask, Flask-CORS, OpenCV, scikit-learn, scikit-image, joblib, numpy, pandas.
- **ML:** Random Forest, SVM, KNN, Gradient Boosting, XGBoost, LightGBM, VotingClassifier (ensemble).
- **Frontend:** React, React Router, Material-UI (MUI), Framer Motion.
- **API:** REST; JSON request/response.

## 4.2 Feature Extraction Implementation

- **File:** `src/image_preprocessing.py`
- **Class:** `RiceImageFeatureExtractor`
- **Methods:** preprocess_image, extract_color_features, extract_texture_features_glcm, extract_texture_features_lbp, extract_size_shape_features, extract_all_features.
- **Output:** Dictionary of 25 feature names and values; same order as `feature_names` for the model.

## 4.3 Model Training Pipeline

- **Script:** `train_all_13_varieties.py`
- **Steps:** Scan `data/images/<variety>/` and optionally prediction history → extract features per image → build DataFrame → train/test split → StandardScaler fit on train → train RF, SVM, KNN, GB, LGBM, XGB, Ensemble → evaluate → save best model, scaler, metadata.

## 4.4 API Implementation

- **File:** `api_server.py`
- **Endpoints:** See Appendix C. Key: POST `/api/predict` (multipart image), GET `/api/prediction-history`, POST `/api/compare`, GET `/api/dashboard-data`, GET/POST `/api/listings`, admin exports, etc.
- **Config:** UPLOAD_FOLDER, PUBLIC_IMAGES_FOLDER, allowed extensions, max file size; model and scaler loaded at startup.

## 4.5 Frontend Implementation

- **App entry:** `rice-ml-dashboard/src/App.js` — Router, theme, drawer menu, routes to pages.
- **Pages:** PredictPage (upload, camera, result, features), ComparePage, HistoryPage, MarketplacePage, DatasetPage, AnalysisPage, AdminPage.
- **API base URL:** `http://localhost:5001/api` (configurable).

---

# Chapter 5 — Testing

## 5.1 Test Strategy

- **Unit:** Feature extraction output shape and range; validation logic (reject blue-dominant, etc.).
- **Integration:** API endpoints with sample images; dashboard pages loading and displaying data.
- **User acceptance:** Upload/camera → correct prediction display; history and compare flows.

## 5.2 Test Cases and Results

| # | Test case | Expected | Result (sample) |
|---|------------|----------|------------------|
| 1 | Upload valid rice image | 200, prediction + top-3 | Pass |
| 2 | Upload non-image file | 400, error message | Pass |
| 3 | Upload non-rice image (e.g. scenery) | 400, NOT_RICE_IMAGE | Pass |
| 4 | GET /api/health | 200, model_loaded true | Pass |
| 5 | GET /api/prediction-history | 200, list of predictions | Pass |
| 6 | Compare two rice images | 200, two predictions | Pass |
| 7 | Dashboard loads all pages | No console errors, data shown | Pass |

*(Fill with your actual test results.)*

## 5.3 Screenshots of Working System

*Insert the following screenshots in order in the appendix (Appendix B).*

1. **Dashboard – Navigation**  
   **[INSERT SCREENSHOT: Sidebar with Predict, Compare, History, Marketplace, Dataset, Analysis, Admin.]**

2. **Predict – Upload**  
   **[INSERT SCREENSHOT: Upload area and buttons before prediction.]**

3. **Predict – Result**  
   **[INSERT SCREENSHOT: Image, predicted variety, confidence, top-3, features table.]**

4. **Predict – Camera**  
   **[INSERT SCREENSHOT: Camera dialog with video and capture button.]**

5. **Compare**  
   **[INSERT SCREENSHOT: Two images with their predictions.]**

6. **History**  
   **[INSERT SCREENSHOT: List of past predictions with thumbnails.]**

7. **Marketplace**  
   **[INSERT SCREENSHOT: Listings and sell form.]**

8. **Dataset**  
   **[INSERT SCREENSHOT: Class distribution and stats.]**

9. **Analysis**  
   **[INSERT SCREENSHOT: Model metrics and charts.]**

10. **Admin**  
    **[INSERT SCREENSHOT: Admin stats and export options.]**

11. **Sample prediction images from project**  
    *(Optional: 1–2 screenshots from `rice-ml-dashboard/public/prediction_images/` or History page showing real prediction thumbnails.)*

---

# Chapter 6 — Conclusion and Future Work

## 6.1 Conclusion

This project implemented a complete **Rice Variety Classification System** for 13 Indian rice varieties. It includes image preprocessing, handcrafted feature extraction, multi-class ML training (with ensemble selection), non-rice validation, a Flask REST API, and a React dashboard with Predict (upload + camera), Compare, History, Marketplace, Dataset, Analysis, and Admin. The system is deployable, interpretable, and suitable for extension with more data and varieties.

## 6.2 Limitations

- Accuracy depends on quality and balance of training data per variety.
- Best results with single-grain or clear-background images.
- Handcrafted features may not capture all visual nuances; deep learning could complement in the future.

## 6.3 Future Enhancements

- Add more rice varieties and augment dataset.
- Integrate a CNN option (transfer learning or small custom CNN).
- Mobile app (React Native or PWA).
- Integration with weighing/labeling devices and barcode/QR.
- Multi-language support in dashboard.

---

# References / Bibliography

1. scikit-learn Documentation. https://scikit-learn.org/
2. OpenCV Documentation. https://docs.opencv.org/
3. Flask Documentation. https://flask.palletsprojects.com/
4. React Documentation. https://react.dev/
5. Material-UI. https://mui.com/
6. Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Trans. Syst. Man Cybern.
7. Haralick, R. M., et al. (1973). Textural features for image classification. IEEE Trans. Syst. Man Cybern.
8. [Add any project-specific or course-recommended references.]

---

# Appendix A — Sample Code Snippets

**Feature extraction call (Python):**
```python
from src.image_preprocessing import RiceImageFeatureExtractor
extractor = RiceImageFeatureExtractor()
features = extractor.extract_all_features(image_array)
feature_vector = np.array([[features.get(n, 0.0) for n in extractor.feature_names]])
```

**API predict (curl):**
```bash
curl -X POST http://localhost:5001/api/predict -F "image=@rice_sample.jpg"
```

**Dashboard API base (React):**
```javascript
const API_BASE_URL = 'http://localhost:5001/api';
fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: formData });
```

---

# Appendix B — Screenshots (All Project Screens)

*Place all project screenshots here in order. Suggested folder: `docs/screenshots/` or `blackbook_screenshots/`.*

| Figure | Description | Filename suggestion |
|--------|-------------|----------------------|
| B.1 | Dashboard – Navigation | 01-dashboard-nav.png |
| B.2 | Predict – Upload | 02-predict-upload.png |
| B.3 | Predict – Result | 03-predict-result.png |
| B.4 | Predict – Camera | 04-predict-camera.png |
| B.5 | Compare page | 05-compare.png |
| B.6 | History page | 06-history.png |
| B.7 | Marketplace page | 07-marketplace.png |
| B.8 | Dataset page | 08-dataset.png |
| B.9 | Analysis page | 09-analysis.png |
| B.10 | Admin page | 10-admin.png |
| B.11 | Sample prediction images | 11-sample-predictions.png |

**Where to get screenshots:**  
- Run backend: `python api_server.py` (port 5001).  
- Run frontend: `cd rice-ml-dashboard && npm start` (port 3000).  
- Open http://localhost:3000 and capture each page.  
- Prediction images are also in: `rice-ml-dashboard/public/prediction_images/`.

---

# Appendix C — API Endpoint List

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/api/health` | Health, model/scaler status, varieties |
| POST | `/api/predict` | Upload image → prediction + top-3 + features |
| POST | `/api/extract-features` | Extract features only |
| POST | `/api/compare` | Compare two images |
| GET | `/api/rice-varieties` | List 13 varieties |
| GET | `/api/dashboard-data` | Dashboard payload (project, dataset, model, history) |
| GET | `/api/prediction-stats` | Aggregate prediction stats |
| GET | `/api/prediction-history` | Full prediction history |
| GET | `/api/images/<filename>` | Serve prediction image |
| POST | `/api/clear-history` | Clear prediction history |
| GET | `/api/listings` | Get marketplace listings |
| POST | `/api/listings` | Create listing |
| POST | `/api/listings/<id>/purchase` | Record purchase |
| GET | `/api/admin/stats` | Admin statistics |
| GET | `/api/admin/export/purchases` | Export purchases CSV |
| GET | `/api/admin/export/listings` | Export listings CSV |

---

# Appendix D — User Manual (Brief)

1. **Start backend:** In project root, activate venv, run `python api_server.py`. Default: http://localhost:5001.
2. **Start frontend:** `cd rice-ml-dashboard`, run `npm start`. Default: http://localhost:3000.
3. **Predict:** Open Predict page → choose “Choose Image” or “Capture from Camera” → after capture/selection, click “Predict” → view result (variety, confidence, top-3, features).
4. **Compare:** Open Compare page → upload two images → view side-by-side predictions.
5. **History:** Open History page → see all past predictions; optional “Clear history.”
6. **Marketplace:** View listings; add listing (variety, price, contact, email); record purchase.
7. **Dataset/Analysis:** View dataset stats and model performance from dashboard data.
8. **Admin:** View stats; export purchases/listings as CSV.

**Supported image types:** PNG, JPG, JPEG, GIF, BMP. Max size: 16 MB. For best results use clear rice grain images on a plain background.

---

*End of Blackbook Content*
