# sign-language-rsa
# SignVision — AI Sign Language Recognition

> **Real-time AI-powered static ASL sign recognition** using MediaPipe hand landmarks, Reptile Search Algorithm (RSA) feature selection, feature engineering, and KNN classification.

## 🌐 Live Demo

**Live Dashboard:**
https://sign-language-rsa-1.onrender.com

**Backend API:**
https://sign-language-rsa.onrender.com

> Open the Live Dashboard, allow camera access, and use your webcam to recognize the supported ASL signs **A, B, C, L, O, V, and Y**. Optional browser-based voice output is also available.

---

## 📌 Overview

**SignVision** is a full-stack computer vision application for real-time recognition of static American Sign Language (ASL) alphabet gestures.

The system captures a hand image from the webcam, detects hand landmarks using MediaPipe, transforms the landmarks into a normalized feature representation, and predicts the sign using a webcam-calibrated machine learning model.

The project also includes a **Reptile Search Algorithm (RSA)** feature-selection pipeline as a research component for reducing the landmark feature space.

### Current supported signs

```text
A · B · C · L · O · V · Y
```

---

## ✨ Key Features

| Feature                      | Description                                       | Status      |
| ---------------------------- | ------------------------------------------------- | ----------- |
| **Real-Time Recognition**    | Webcam-based static ASL recognition               | ✅ Completed |
| **MediaPipe Hand Detection** | Detects 21 hand landmarks                         | ✅ Completed |
| **Feature Engineering**      | Converts landmarks into 63 normalized features    | ✅ Completed |
| **RSA Feature Selection**    | Selects a reduced subset of landmark features     | ✅ Completed |
| **KNN Live Classifier**      | Webcam-calibrated classifier for live recognition | ✅ Completed |
| **Voice Output**             | Speaks recognized signs in the browser            | ✅ Completed |
| **Prediction History**       | Displays recent recognized classes                | ✅ Completed |
| **React Dashboard**          | Futuristic real-time user interface               | ✅ Completed |
| **FastAPI Backend**          | REST API for image prediction                     | ✅ Completed |
| **Docker Support**           | Containerized backend                             | ✅ Completed |
| **Cloud Deployment**         | Frontend and backend deployed online              | ✅ Completed |

---

## 🎥 Live Recognition Pipeline

```text
                    ┌────────────────────┐
                    │      Webcam        │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │     MediaPipe      │
                    │   Hand Detection   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ 21 Hand Landmarks  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Feature            │
                    │ Normalization      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ 63 Landmark       │
                    │ Features           │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   StandardScaler   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │        KNN         │
                    │    Classifier      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   ASL Prediction   │
                    │ A B C L O V Y      │
                    └─────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             ┌─────────────┐     ┌─────────────┐
             │ Dashboard   │     │ Voice Output│
             └─────────────┘     └─────────────┘
```

---

## 🔬 Reptile Search Algorithm

The project includes the **Reptile Search Algorithm (RSA)** as a feature-selection research component.

The original MediaPipe representation contains:

```text
63 features
```

RSA selected:

```text
35 features
```

Feature reduction:

```text
63 → 35
44.44% reduction
```

### RSA workflow

```text
63 Landmark Features
        │
        ▼
Reptile Search Algorithm
        │
        ▼
Feature Evaluation
        │
        ▼
35 Selected Features
```

The live production classifier currently uses the **full 63-feature representation with KNN**, because webcam-specific calibration produced better practical live recognition with that configuration.

RSA remains an independently implemented research component for feature selection and dimensionality reduction.

---

## 🤖 Machine Learning

### Live model

```text
MediaPipe
   ↓
63 normalized features
   ↓
StandardScaler
   ↓
KNN
   ↓
A / B / C / L / O / V / Y
```

### Webcam calibration

The live model was calibrated using webcam landmark samples for the seven supported classes.

Current calibration training set:

```text
Training samples: 200
Classes: 7
Features: 63
```

The additional calibration data specifically improved the live recognition of the more difficult classes during development.

---

## 🧠 Computer Vision

MediaPipe is used to detect:

```text
21 hand landmarks
```

Each landmark contains:

```text
X coordinate
Y coordinate
Z coordinate
```

Therefore:

```text
21 × 3 = 63 features
```

The landmarks are normalized relative to the wrist and transformed into a canonical hand coordinate system to reduce sensitivity to translation, scale, and hand orientation.

---

## 🏗️ System Architecture

```text
                         SignVision
                             │
             ┌───────────────┴────────────────┐
             │                                │
             ▼                                ▼
       React Frontend                    FastAPI Backend
             │                                │
             │ HTTPS                         │
             └──────────────► /predict ◄──────┘
                                              │
                                              ▼
                                        OpenCV Image
                                              │
                                              ▼
                                          MediaPipe
                                              │
                                              ▼
                                      Hand Landmarks
                                              │
                                              ▼
                                      Feature Extractor
                                              │
                                              ▼
                                        63 Features
                                              │
                                              ▼
                                       StandardScaler
                                              │
                                              ▼
                                             KNN
                                              │
                                              ▼
                                       Sign Prediction
                                              │
                                              ▼
                                         JSON Response
```

---

## 🖥️ Frontend

The frontend is built using:

* React
* Vite
* JavaScript
* CSS
* Browser MediaDevices API
* Web Speech API

### Dashboard sections

| Section                    | Purpose                                       |
| -------------------------- | --------------------------------------------- |
| **Live Camera**            | Displays webcam feed                          |
| **Current Prediction**     | Shows the recognized sign                     |
| **Hand Detection**         | Indicates whether a hand is detected          |
| **Recent Signs**           | Shows recent predictions                      |
| **Supported Signs**        | Displays the seven supported classes          |
| **System Pipeline**        | Explains the live ML pipeline                 |
| **RSA Research Component** | Shows the feature-selection research pipeline |
| **Voice Control**          | Enables/disables speech output                |

---

## ⚙️ Backend

The backend is implemented using:

* Python
* FastAPI
* Uvicorn
* OpenCV
* MediaPipe
* NumPy
* Pandas
* scikit-learn
* Joblib

### API endpoints

| Method | Endpoint      | Description                           |
| ------ | ------------- | ------------------------------------- |
| `GET`  | `/`           | API information                       |
| `GET`  | `/health`     | Service health and model metadata     |
| `GET`  | `/model-info` | Model and RSA information             |
| `POST` | `/predict`    | Predict a sign from an uploaded image |
| `GET`  | `/docs`       | FastAPI Swagger documentation         |

### Example prediction response

```json
{
  "hand_detected": true,
  "prediction": "A",
  "confidence": 0.92,
  "features": 63,
  "model": "KNN"
}
```

---

## 🌐 Deployment

### Frontend

The React frontend is deployed as a Render Static Site.

**Live Dashboard:**

https://sign-language-rsa-1.onrender.com

### Backend

The FastAPI backend is deployed as a Docker web service.

**Backend API:**

https://sign-language-rsa.onrender.com

### Container Architecture

```text
Render
  │
  ├── Frontend
  │     └── React + Vite
  │
  └── Backend
        └── Docker
             └── FastAPI
```

---

## 🛠️ Technology Stack

### Frontend

```text
React
Vite
JavaScript
CSS
Web Speech API
MediaDevices API
```

### Backend

```text
Python
FastAPI
Uvicorn
OpenCV
MediaPipe
NumPy
Pandas
Joblib
```

### Machine Learning

```text
scikit-learn
KNN
RBF-SVM
StandardScaler
Feature Engineering
Reptile Search Algorithm
```

### Deployment

```text
Docker
GitHub
Render
```

---

## 📂 Project Structure

```text
sign-language-rsa/
│
├── backend/
│   ├── ai/
│   │   ├── asl_inference.py
│   │   ├── inference.py
│   │   ├── model.py
│   │   └── preprocessing.py
│   │
│   ├── api/
│   │   ├── analytics.py
│   │   ├── prediction.py
│   │   └── training.py
│   │
│   ├── models/
│   │   ├── asl_landmark_baseline_svm.joblib
│   │   ├── asl_model_comparison.joblib
│   │   ├── asl_rsa_selected_indices.joblib
│   │   ├── asl_rsa_svm.joblib
│   │   └── session_live_knn.joblib
│   │
│   ├── rsa/
│   │   ├── fitness.py
│   │   ├── optimizer.py
│   │   ├── public_fitness.py
│   │   ├── public_optimizer.py
│   │   └── reptile_search.py
│   │
│   ├── vision/
│   │   ├── feature_extractor.py
│   │   ├── hand_detector.py
│   │   └── public_live_prediction.py
│   │
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── results/
│   ├── accuracy_comparison.png
│   ├── baseline_confusion_matrix.png
│   ├── feature_reduction.png
│   └── rsa_confusion_matrix.png
│
├── scripts/
│   ├── collect_webcam_session.py
│   ├── evaluate_session_models.py
│   ├── train_asl_baseline.py
│   ├── train_asl_rsa.py
│   └── ...
│
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/prudhvimareedu/sign-language-rsa.git
cd sign-language-rsa
```

---

### 2. Backend setup

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r backend/requirements.txt
```

Start FastAPI:

```powershell
python -m uvicorn backend.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

### 3. Frontend setup

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

Allow browser camera access when prompted.

---

## 🐳 Docker

The backend includes a Dockerfile for deployment.

Build:

```bash
docker build -t signvision-backend .
```

Run:

```bash
docker run -p 8000:8000 signvision-backend
```

The API will be available at:

```text
http://localhost:8000
```

---

## 📊 Research Experiments

The repository contains separate scripts for model experimentation and RSA evaluation.

### Public landmark baseline

```bash
python scripts/train_asl_baseline.py
```

### RSA feature selection

```bash
python scripts/train_asl_rsa.py
```

### Webcam session evaluation

```bash
python scripts/evaluate_session_models.py
```

### Webcam data collection

```bash
python scripts/collect_webcam_session.py
```

These scripts document the experimental side of the project while the production application uses the finalized webcam-calibrated KNN model.

---

## 📈 Results

### RSA feature reduction

```text
Original landmark features: 63
RSA-selected features:      35
Feature reduction:          44.44%
```

### Live application

The final live system supports:

```text
A
B
C
L
O
V
Y
```

The final model was selected and calibrated using webcam-specific data, followed by direct browser testing of the supported classes.

---

## ⚠️ Limitations

This project focuses on **static ASL alphabet gesture recognition**, not full continuous sign-language translation.

Current supported signs:

```text
A, B, C, L, O, V, Y
```

Recognition can vary based on:

* Lighting conditions
* Camera quality
* Hand orientation
* Distance from camera
* Background conditions
* Individual hand shape
* Hand position within the frame

The current system is intended as a practical prototype and research-oriented computer vision application rather than a complete sign-language translation system.

---

## 🔮 Future Improvements

* Extend recognition to the complete ASL alphabet
* Add dynamic gesture recognition
* Recognize continuous sign sequences
* Introduce temporal models such as LSTM or Transformer architectures
* Build a larger multi-person webcam dataset
* Improve confidence calibration
* Add multilingual speech output
* Add user-specific calibration
* Add model monitoring and analytics
* Improve robustness under changing lighting and backgrounds
* Deploy a larger-scale production inference service

---

## 🎯 Project Highlights

SignVision combines:

```text
Computer Vision
        +
Feature Engineering
        +
Reptile Search Algorithm
        +
Machine Learning
        +
FastAPI
        +
React
        +
Docker
        +
Cloud Deployment
```

The result is a complete end-to-end AI application that connects webcam perception, machine learning inference, browser interaction, and cloud deployment.

---

## 👨‍💻 Author

**Prudhvi Raju Mareedu**

B.Tech — Information Technology

GitHub:
https://github.com/prudhvimareedu

Project Repository:
https://github.com/prudhvimareedu/sign-language-rsa

Live Dashboard:
https://sign-language-rsa-1.onrender.com

Backend API:
https://sign-language-rsa.onrender.com

---

## 📄 License

This project is intended for educational, portfolio, and research purposes.
