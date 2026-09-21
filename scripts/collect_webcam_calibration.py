import cv2
import csv
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.vision.hand_detector import HandDetector
from backend.vision.feature_extractor import FeatureExtractor


CLASSES = ["A", "B", "C", "L", "O", "V", "Y"]
SAMPLES_PER_CLASS = 50

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "webcam_calibration.csv"
)


def main():

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
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

    print("========================================")
    print("WEBCAM CALIBRATION DATASET")
    print("========================================")
    print()
    print("Classes:", ", ".join(CLASSES))
    print("Samples per class:", SAMPLES_PER_CLASS)
    print("Total samples:", len(CLASSES) * SAMPLES_PER_CLASS)
    print()
    print("Keep your hand clearly visible.")
    print("Move it slightly between samples.")
    print("Press Q to quit.")
    print()

    header = [f"f{i}" for i in range(63)] + ["label"]

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)
        writer.writerow(header)

        for label in CLASSES:

            print()
            print(
                f"Get ready for sign: {label}"
            )

            for countdown in [3, 2, 1]:
                print(countdown)
                time.sleep(1)

            collected = 0

            while collected < SAMPLES_PER_CLASS:

                success, frame = camera.read()

                if not success:
                    continue

                results = detector.detect(frame)

                if results.multi_hand_landmarks:

                    hand_landmarks = (
                        results.multi_hand_landmarks[0]
                    )

                    features = extractor.extract(
                        hand_landmarks
                    )

                    row = (
                        features.tolist()
                        + [label]
                    )

                    writer.writerow(row)
                    collected += 1

                display = frame.copy()

                display = detector.draw_landmarks(
                    display,
                    results
                )

                cv2.rectangle(
                    display,
                    (10, 10),
                    (520, 100),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    display,
                    f"Sign: {label}",
                    (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"Samples: {collected}/{SAMPLES_PER_CLASS}",
                    (25, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Webcam Calibration",
                    display
                )

                key = cv2.waitKey(30) & 0xFF

                if key == ord("q") or key == ord("Q"):
                    camera.release()
                    cv2.destroyAllWindows()
                    print()
                    print("Calibration stopped.")
                    return

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("========================================")
    print("CALIBRATION COMPLETE")
    print("========================================")
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()