import cv2
import sys
import os
import joblib
import numpy as np
from collections import Counter


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

SIGNS = [
    "A",
    "B",
    "C",
    "L",
    "O",
    "V",
    "Y"
]

FRAMES_PER_SIGN = 60


def get_prediction(
    frame,
    detector,
    extractor,
    model,
    selected_indices
):

    results = detector.detect(frame)

    if not results.multi_hand_landmarks:
        return None, frame

    hand_landmarks = results.multi_hand_landmarks[0]

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

    prediction = model.predict(
        selected_features
    )[0]

    frame = detector.draw_landmarks(
        frame,
        results
    )

    return str(prediction), frame


def main():

    print("========================================")
    print("FINAL SCALED RSA LIVE EVALUATION")
    print("========================================")

    # -------------------------------------
    # Load final model
    # -------------------------------------

    if not os.path.exists(MODEL_PATH):
        print()
        print("ERROR: Final model not found:")
        print(MODEL_PATH)
        return

    if not os.path.exists(INDICES_PATH):
        print()
        print("ERROR: RSA feature index file not found:")
        print(INDICES_PATH)
        return

    model = joblib.load(
        MODEL_PATH
    )

    selected_indices = joblib.load(
        INDICES_PATH
    )

    detector = HandDetector()
    extractor = FeatureExtractor()

    print()
    print("Final model loaded successfully.")
    print(
        "RSA selected features:",
        len(selected_indices)
    )

    # -------------------------------------
    # Open webcam
    # -------------------------------------

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        print()
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
    print("You will test these signs:")
    print("A  B  C  L  O  V  Y")
    print()
    print("For every sign:")
    print("1. Show the requested sign.")
    print("2. Press SPACE.")
    print("3. Hold the same sign steadily.")
    print("4. Wait until the capture finishes.")
    print()
    print("Press Q to quit.")
    print()

    results_summary = []

    # -------------------------------------
    # Test every sign
    # -------------------------------------

    for expected in SIGNS:

        print()
        print("----------------------------------------")
        print(
            f"GET READY: {expected}"
        )
        print("----------------------------------------")
        print(
            "Show the sign and press SPACE."
        )

        # ---------------------------------
        # Wait for SPACE
        # ---------------------------------

        while True:

            success, frame = camera.read()

            if not success:
                continue

            display = frame.copy()

            cv2.rectangle(
                display,
                (10, 10),
                (650, 115),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                display,
                f"Expected sign: {expected}",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                "Press SPACE to start",
                (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Final RSA Live Evaluation",
                display
            )

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == ord("Q"):
                camera.release()
                cv2.destroyAllWindows()
                print()
                print("Evaluation cancelled.")
                return

            if key == 32:
                break

        # ---------------------------------
        # Capture 60 frames
        # ---------------------------------

        predictions = []

        frame_counter = 0

        while frame_counter < FRAMES_PER_SIGN:

            success, frame = camera.read()

            if not success:
                continue

            prediction, display = get_prediction(
                frame,
                detector,
                extractor,
                model,
                selected_indices
            )

            if prediction is not None:
                predictions.append(
                    prediction
                )

            frame_counter += 1

            if prediction is None:
                current_prediction = "No hand"
            else:
                current_prediction = prediction

            cv2.rectangle(
                display,
                (10, 10),
                (680, 145),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                display,
                f"Expected: {expected}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Current: {current_prediction}",
                (25, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Frame: {frame_counter}/{FRAMES_PER_SIGN}",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Final RSA Live Evaluation",
                display
            )

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == ord("Q"):
                camera.release()
                cv2.destroyAllWindows()
                print()
                print("Evaluation cancelled.")
                return

        # ---------------------------------
        # Majority vote
        # ---------------------------------

        if predictions:

            counts = Counter(
                predictions
            )

            majority_prediction = (
                counts.most_common(1)[0][0]
            )

            correct = (
                majority_prediction == expected
            )

            top_predictions = (
                counts.most_common(5)
            )

        else:

            majority_prediction = "No hand"
            correct = False
            top_predictions = []

        # ---------------------------------
        # Store result
        # ---------------------------------

        results_summary.append(
            {
                "expected": expected,
                "predicted": majority_prediction,
                "correct": correct,
                "frames": len(predictions)
            }
        )

        print()
        print(
            f"Expected: {expected}"
        )

        print(
            f"Predicted: {majority_prediction}"
        )

        print(
            f"Frames detected: {len(predictions)}"
        )

        print(
            f"Correct: {correct}"
        )

        if top_predictions:

            print(
                "Top predictions:"
            )

            for label, count in top_predictions:

                percentage = (
                    count
                    / len(predictions)
                    * 100
                )

                print(
                    f"  {label}: "
                    f"{count} "
                    f"({percentage:.1f}%)"
                )

    # -------------------------------------
    # Close webcam
    # -------------------------------------

    camera.release()
    cv2.destroyAllWindows()

    # -------------------------------------
    # Calculate live accuracy
    # -------------------------------------

    correct_count = sum(
        item["correct"]
        for item in results_summary
    )

    total_signs = len(
        results_summary
    )

    if total_signs > 0:

        live_accuracy = (
            correct_count
            / total_signs
            * 100
        )

    else:

        live_accuracy = 0.0

    # -------------------------------------
    # Final report
    # -------------------------------------

    print()
    print("========================================")
    print("LIVE EVALUATION RESULT")
    print("========================================")

    for item in results_summary:

        status = (
            "CORRECT"
            if item["correct"]
            else "WRONG"
        )

        print(
            f"{item['expected']} -> "
            f"{item['predicted']} "
            f"| {status}"
        )

    print()
    print(
        f"Correct signs: "
        f"{correct_count}/{total_signs}"
    )

    print(
        f"Live sign accuracy: "
        f"{live_accuracy:.2f}%"
    )

    print()
    print("========================================")
    print("EVALUATION FINISHED")
    print("========================================")


if __name__ == "__main__":
    main()