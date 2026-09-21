import os
import sys
import time
import cv2

# Add project root
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
from backend.ai.inference import SignLanguageInference


detector = HandDetector()
extractor = FeatureExtractor()
inference = SignLanguageInference()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    sys.exit()

print("========================================")
print("REAL-TIME SIGN LANGUAGE PREDICTION")
print("========================================")
print("Camera opened successfully.")
print("Show A, B, C, D... to the camera.")
print("Press Q inside the camera window to stop.")
print("========================================")


last_prediction = "---"
last_prediction_time = 0


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    results = detector.detect(frame)

    frame = detector.draw_landmarks(
        frame,
        results
    )

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        features = extractor.extract(
            hand_landmarks
        )

        # Predict only once every 0.2 seconds
        current_time = time.time()

        if current_time - last_prediction_time >= 0.2:

            last_prediction = inference.predict(
                features
            )

            last_prediction_time = current_time

            print(
                "Prediction:",
                last_prediction
            )

        cv2.putText(
            frame,
            f"Prediction: {last_prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
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
        "RSA Sign Language Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("Prediction test stopped.")