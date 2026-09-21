import os
import sys
import cv2
import joblib
import numpy as np


# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# =========================================================
# Hand detector
# =========================================================

from hand_detector import HandDetector


# =========================================================
# Model paths
# =========================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "public_rsa_svm.joblib"
)


FEATURE_INDEX_PATH = os.path.join(
    MODEL_DIR,
    "public_rsa_selected_indices.joblib"
)


# =========================================================
# Load model
# =========================================================

print("Loading RSA + SVM model...")

model = joblib.load(
    MODEL_PATH
)

selected_indices = np.asarray(
    joblib.load(
        FEATURE_INDEX_PATH
    ),
    dtype=np.int64
)

print(
    "Selected HOG features:",
    len(selected_indices)
)


# =========================================================
# HOG configuration
# MUST match training
# =========================================================

hog = cv2.HOGDescriptor(
    (28, 28),   # window size
    (14, 14),   # block size
    (7, 7),     # block stride
    (7, 7),     # cell size
    9           # orientations
)


# =========================================================
# Sign Language MNIST label mapping
# 9 is missing because J requires movement
# =========================================================

LABEL_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y"
}


# =========================================================
# Initialize hand detector
# =========================================================

detector = HandDetector()


# =========================================================
# Open webcam
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open camera.")
    sys.exit(1)


print()
print("========================================")
print("RSA SIGN LANGUAGE LIVE PREDICTION")
print("========================================")
print("Show A, B, C, D... to the camera.")
print("Prediction is printed in PowerShell.")
print("Press Q to stop.")
print("========================================")
print()


# =========================================================
# Main loop
# =========================================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "ERROR: Could not read webcam frame."
        )

        break


    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )


    # Detect hand
    results = detector.detect(
        frame
    )


    # Draw landmarks
    frame = detector.draw_landmarks(
        frame,
        results
    )


    prediction_text = "---"


    # =====================================================
    # If a hand is detected
    # =====================================================

    if results.multi_hand_landmarks:

        hand = (
            results.multi_hand_landmarks[0]
        )


        height, width = frame.shape[:2]


        # -------------------------------------------------
        # Convert normalized MediaPipe coordinates
        # into image pixels
        # -------------------------------------------------

        x_values = [
            int(point.x * width)
            for point in hand.landmark
        ]

        y_values = [
            int(point.y * height)
            for point in hand.landmark
        ]


        # -------------------------------------------------
        # Create bounding box
        # -------------------------------------------------

        padding = 30

        x_min = max(
            min(x_values) - padding,
            0
        )

        x_max = min(
            max(x_values) + padding,
            width
        )

        y_min = max(
            min(y_values) - padding,
            0
        )

        y_max = min(
            max(y_values) + padding,
            height
        )


        # -------------------------------------------------
        # Crop hand
        # -------------------------------------------------

        roi = frame[
            y_min:y_max,
            x_min:x_max
        ]


        if (
            roi.size > 0
            and roi.shape[0] > 5
            and roi.shape[1] > 5
        ):


            # -------------------------------------------------
            # Convert to grayscale
            # -------------------------------------------------

            gray = cv2.cvtColor(
                roi,
                cv2.COLOR_BGR2GRAY
            )


            # -------------------------------------------------
            # Resize to 28x28
            # -------------------------------------------------

            gray = cv2.resize(
                gray,
                (28, 28),
                interpolation=cv2.INTER_AREA
            )


            # -------------------------------------------------
            # HOG features
            # -------------------------------------------------

            features = hog.compute(
                gray
            )


            features = features.flatten()


            # -------------------------------------------------
            # RSA-selected features
            # -------------------------------------------------

            selected_features = features[
                selected_indices
            ]


            selected_features = (
                selected_features.reshape(
                    1,
                    -1
                )
            )


            # -------------------------------------------------
            # Predict
            # -------------------------------------------------

            prediction = model.predict(
                selected_features
            )[0]


            prediction_text = LABEL_MAP.get(
                int(prediction),
                "?"
            )


            # -------------------------------------------------
            # Print prediction to PowerShell
            # -------------------------------------------------

            print(
                "Prediction:",
                prediction_text
            )


            # -------------------------------------------------
            # Draw prediction on camera
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Prediction: {prediction_text}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )


            # -------------------------------------------------
            # Draw hand bounding box
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )


        else:

            cv2.putText(
                frame,
                "Invalid hand crop",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
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


    # =====================================================
    # Show camera
    # =====================================================

    cv2.imshow(
        "RSA Sign Language Recognition",
        frame
    )


    # =====================================================
    # Quit
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =========================================================
# Cleanup
# =========================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Prediction test stopped.")