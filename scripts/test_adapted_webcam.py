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
    "asl_rsa_webcam_adapted_svm.joblib"
)

SELECTED_INDICES_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "asl_rsa_selected_indices.joblib"
)


def main():

    print("========================================")
    print("ADAPTED RSA LIVE WEBCAM TEST")
    print("========================================")

    model = joblib.load(
        MODEL_PATH
    )

    selected_indices = joblib.load(
        SELECTED_INDICES_PATH
    )

    detector = HandDetector()
    extractor = FeatureExtractor()

    print()
    print(
        "Model loaded successfully."
    )

    print(
        "RSA features:",
        len(selected_indices)
    )

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    prediction_history = deque(
        maxlen=15
    )

    last_smoothed = None

    print()
    print("Webcam started.")
    print()
    print("Test these signs:")
    print("A  B  C  L  O  V  Y")
    print()
    print("Hold each sign for 3 seconds.")
    print("Press Q to quit.")
    print()

    while True:

        success, frame = camera.read()

        if not success:
            print(
                "ERROR: Could not read webcam frame."
            )
            break

        results = detector.detect(
            frame
        )

        raw_prediction = "No hand"
        smoothed_prediction = "No hand"

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
            ]

            selected_features = (
                selected_features.reshape(
                    1,
                    -1
                )
            )

            raw_prediction = str(
                model.predict(
                    selected_features
                )[0]
            )

            prediction_history.append(
                raw_prediction
            )

            counts = Counter(
                prediction_history
            )

            smoothed_prediction = (
                counts.most_common(1)[0][0]
            )

        else:

            prediction_history.clear()

        if (
            smoothed_prediction
            != last_smoothed
        ):

            print(
                f"Raw: {raw_prediction} | "
                f"Smoothed: {smoothed_prediction}"
            )

            last_smoothed = (
                smoothed_prediction
            )

        frame = detector.draw_landmarks(
            frame,
            results
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (570, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Raw: {raw_prediction}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Smoothed: {smoothed_prediction}",
            (25, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Hold sign steady",
            (25, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Adapted RSA ASL",
            frame
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("========================================")
    print("ADAPTED LIVE TEST FINISHED")
    print("========================================")


if __name__ == "__main__":
    main()