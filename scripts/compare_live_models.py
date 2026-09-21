import cv2
import sys
import os
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.vision.hand_detector import HandDetector
from backend.vision.feature_extractor import FeatureExtractor


def main():

    print("========================================")
    print("BASELINE vs RSA LIVE COMPARISON")
    print("========================================")

    baseline_path = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models",
        "asl_landmark_baseline_svm.joblib"
    )

    rsa_path = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models",
        "asl_rsa_svm.joblib"
    )

    indices_path = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models",
        "asl_rsa_selected_indices.joblib"
    )

    baseline_model = joblib.load(
        baseline_path
    )

    rsa_model = joblib.load(
        rsa_path
    )

    selected_indices = joblib.load(
        indices_path
    )

    detector = HandDetector()
    extractor = FeatureExtractor()

    print()
    print("Baseline model: 63 features")
    print(
        "RSA model:",
        len(selected_indices),
        "features"
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

    print()
    print("Webcam started.")
    print("Show one sign at a time.")
    print("Hold each sign for 3 seconds.")
    print("Test: A, B, C, L, O, V, Y")
    print("Press Q to quit.")
    print()

    last_baseline = None
    last_rsa = None

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        results = detector.detect(frame)

        baseline_prediction = "No hand"
        rsa_prediction = "No hand"

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

            features_2d = features.reshape(
                1,
                -1
            )

            # -----------------------------
            # BASELINE
            # -----------------------------

            baseline_prediction = (
                baseline_model.predict(
                    features_2d
                )[0]
            )

            # -----------------------------
            # RSA
            # -----------------------------

            rsa_features = features[
                selected_indices
            ]

            rsa_features = rsa_features.reshape(
                1,
                -1
            )

            rsa_prediction = (
                rsa_model.predict(
                    rsa_features
                )[0]
            )

        if (
            baseline_prediction != last_baseline
            or rsa_prediction != last_rsa
        ):

            print(
                "Baseline:",
                baseline_prediction,
                "| RSA:",
                rsa_prediction
            )

            last_baseline = baseline_prediction
            last_rsa = rsa_prediction

        frame = detector.draw_landmarks(
            frame,
            results
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (560, 120),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Baseline: {baseline_prediction}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"RSA: {rsa_prediction}",
            (25, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Baseline vs RSA",
            frame
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("Comparison test finished.")


if __name__ == "__main__":
    main()