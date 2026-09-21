import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, PROJECT_ROOT)

from hand_detector import HandDetector
from feature_extractor import FeatureExtractor


# -----------------------------
# Load model and scaler
# -----------------------------

models_dir = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)

model = joblib.load(
    os.path.join(
        models_dir,
        "baseline_model.joblib"
    )
)

scaler = joblib.load(
    os.path.join(
        models_dir,
        "feature_scaler.joblib"
    )
)

feature_names = []

for i in range(21):
    feature_names.append(f"x_{i}")
    feature_names.append(f"y_{i}")
    feature_names.append(f"z_{i}")


# -----------------------------
# Initialize vision
# -----------------------------

detector = HandDetector()
extractor = FeatureExtractor()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    sys.exit()


print("========================================")
print("BASELINE LIVE PREDICTION TEST")
print("========================================")
print("Show ONE trained sign.")
print("Press Q to quit.")
print("========================================")


while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    results = detector.detect(frame)

    frame = detector.draw_landmarks(
        frame,
        results
    )

    if results.multi_hand_landmarks:

        landmarks = results.multi_hand_landmarks[0]

        features = extractor.extract(
            landmarks
        )

        features = np.asarray(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        features_df = pd.DataFrame(
            features,
            columns=feature_names
        )

        scaled = scaler.transform(
            features_df
        )

        prediction = model.predict(
            scaled
        )[0]

        cv2.putText(
            frame,
            f"Baseline: {prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        print(
            "Prediction:",
            prediction
        )

    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Baseline Sign Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()