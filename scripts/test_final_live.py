import cv2
import sys
import os
import joblib
import numpy as np
from collections import Counter, deque

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.vision.hand_detector import HandDetector
from backend.vision.feature_extractor import FeatureExtractor


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "final_scaled_rsa_svm.joblib"
)

INDICES_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "asl_rsa_selected_indices.joblib"
)


def main():

    print("========================================")
    print("FINAL SCALED RSA LIVE TEST")
    print("========================================")

    model = joblib.load(
        MODEL_PATH
    )

    selected_indices = joblib.load(
        INDICES_PATH
    )

    detector = HandDetector()
    extractor = FeatureExtractor()

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    history = deque(
        maxlen=15
    )

    last_prediction = None

    print()
    print("Model loaded.")
    print("RSA features:", len(selected_indices))
    print()
    print("Test: A B C L O V Y")
    print("Hold each sign for 3 seconds.")
    print("Press Q to quit.")
    print()

    while True:

        success, frame = camera.read()

        if not success:
            break

        results = detector.detect(frame)

        raw_prediction = "No hand"
        prediction = "No hand"

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            features = extractor.extract(
                hand_landmarks
            )

            features = np.asarray(
                features,
                dtype=np.float32
            )

            selected_features = features[
                selected_indices
            ].reshape(1, -1)

            raw_prediction = str(
                model.predict(
                    selected_features
                )[0]
            )

            history.append(
                raw_prediction
            )

            prediction = Counter(
                history
            ).most_common(1)[0][0]

        else:

            history.clear()

        if prediction != last_prediction:

            print(
                f"Raw: {raw_prediction} | "
                f"Smoothed: {prediction}"
            )

            last_prediction = prediction

        frame = detector.draw_landmarks(
            frame,
            results
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (600, 120),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Raw: {raw_prediction}",
            (25, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "A B C L O V Y",
            (25, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Final Scaled RSA ASL",
            frame
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("========================================")
    print("FINAL LIVE TEST FINISHED")
    print("========================================")


if __name__ == "__main__":
    main()