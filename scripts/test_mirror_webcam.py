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


def predict_frame(frame, detector, extractor, model):

    results = detector.detect(frame)

    if not results.multi_hand_landmarks:
        return "No hand", frame

    hand_landmarks = results.multi_hand_landmarks[0]

    features = extractor.extract(
        hand_landmarks
    )

    features = np.asarray(
        features,
        dtype=np.float32
    ).reshape(1, -1)

    prediction = model.predict(
        features
    )[0]

    frame = detector.draw_landmarks(
        frame,
        results
    )

    return str(prediction), frame


def main():

    model_path = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models",
        "asl_landmark_baseline_svm.joblib"
    )

    model = joblib.load(model_path)

    detector_original = HandDetector()
    detector_mirror = HandDetector()

    extractor = FeatureExtractor()

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("========================================")
    print("ORIGINAL vs MIRRORED TEST")
    print("========================================")
    print()
    print("Show one sign at a time.")
    print("Test A, B, C, L, O, V, Y")
    print("Hold each for about 3 seconds.")
    print("Press Q to quit.")
    print()

    last_original = None
    last_mirror = None

    while True:

        success, frame = camera.read()

        if not success:
            break

        original_frame = frame.copy()

        mirror_frame = cv2.flip(
            frame,
            1
        )

        original_prediction, original_frame = predict_frame(
            original_frame,
            detector_original,
            extractor,
            model
        )

        mirror_prediction, mirror_frame = predict_frame(
            mirror_frame,
            detector_mirror,
            extractor,
            model
        )

        if (
            original_prediction != last_original
            or mirror_prediction != last_mirror
        ):

            print(
                f"Original: {original_prediction} | "
                f"Mirrored: {mirror_prediction}"
            )

            last_original = original_prediction
            last_mirror = mirror_prediction

        cv2.rectangle(
            original_frame,
            (10, 10),
            (500, 110),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            original_frame,
            f"Original: {original_prediction}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.putText(
            original_frame,
            f"Mirrored: {mirror_prediction}",
            (25, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Original vs Mirrored",
            original_frame
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("Mirror test finished.")


if __name__ == "__main__":
    main()