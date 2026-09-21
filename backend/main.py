from pathlib import Path
import cv2
import joblib
import mediapipe as mp
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.vision.feature_extractor import FeatureExtractor


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "session_live_knn.joblib"


# =========================================================
# LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

feature_extractor = FeatureExtractor()


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands

image_hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="SignVision API",
    version="1.0.0",
    description="Webcam-based ASL sign language recognition API",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MODEL METADATA
# =========================================================

LIVE_MODEL = "63-feature StandardScaler + KNN"
LIVE_FEATURES = 63
TRAINING_SAMPLES = 200

RSA_FEATURES = 35
RSA_REDUCTION = 44.44

CLASSES = [
    "A",
    "B",
    "C",
    "L",
    "O",
    "V",
    "Y",
]


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "SignVision API is running",
        "live_model": LIVE_MODEL,
        "classes": CLASSES,
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "live_model": LIVE_MODEL,
        "live_features": LIVE_FEATURES,
        "training_samples": TRAINING_SAMPLES,
        "classes": CLASSES,
        "rsa_features": RSA_FEATURES,
        "rsa_feature_reduction_percent": RSA_REDUCTION,
    }


# =========================================================
# MODEL INFO
# =========================================================

@app.get("/model-info")
def model_info():
    return {
        "live_model": LIVE_MODEL,
        "live_features": LIVE_FEATURES,
        "training_samples": TRAINING_SAMPLES,
        "classes": CLASSES,
        "rsa_features": RSA_FEATURES,
        "rsa_feature_reduction_percent": RSA_REDUCTION,
        "research_note": (
            "RSA reduced the original 63-dimensional "
            "landmark representation to 35 features."
        ),
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        image_array = np.frombuffer(
            contents,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode uploaded image.",
            )

        # -------------------------------------------------
        # MediaPipe
        # -------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = image_hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return {
                "hand_detected": False,
                "prediction": "No hand",
                "confidence": 0.0,
            }

        # -------------------------------------------------
        # Extract first hand
        # -------------------------------------------------

        hand_landmarks = results.multi_hand_landmarks[0]

        features = feature_extractor.extract(
            hand_landmarks
        )

        if features is None:
            return {
                "hand_detected": False,
                "prediction": "Unknown",
                "confidence": 0.0,
            }

        features = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, -1)

        # -------------------------------------------------
        # Validate feature count
        # -------------------------------------------------

        if features.shape[1] != LIVE_FEATURES:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Expected {LIVE_FEATURES} features, "
                    f"got {features.shape[1]}."
                ),
            )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(features)[0]

        confidence = 0.0

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            confidence = float(
                np.max(probabilities)
            )

        return {
            "hand_detected": True,
            "prediction": str(prediction),
            "confidence": round(confidence, 4),
            "features": LIVE_FEATURES,
            "model": "KNN",
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("Prediction error:", exc)

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )