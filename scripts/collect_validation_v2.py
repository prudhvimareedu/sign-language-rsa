import os
import sys
import csv
import cv2

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.vision.hand_detector import HandDetector
from backend.vision.feature_extractor import FeatureExtractor


OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "validation.csv"
)

SAMPLES_PER_SIGN = 50

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


detector = HandDetector()
extractor = FeatureExtractor()


file = open(
    OUTPUT_FILE,
    "w",
    newline=""
)

writer = csv.writer(file)

header = []

for i in range(21):
    header.append(f"x_{i}")
    header.append(f"y_{i}")
    header.append(f"z_{i}")

header.append("label")

writer.writerow(header)


print()
print("========================================")
print("CLEAN V2 VALIDATION COLLECTOR")
print("========================================")
print("Collecting fresh A, B, C samples")
print("50 samples per sign")
print("Type EXIT to stop")
print("========================================")


while True:

    label = input(
        "\nEnter sign label: "
    ).strip().upper()

    if label == "EXIT":
        break

    if label not in ["A", "B", "C"]:
        print("Enter only A, B, or C.")
        continue

    print()
    print(f"Prepare sign: {label}")
    print("This is a NEW validation session.")
    print("Change the hand position slightly.")
    print("Change the distance slightly.")
    print("Keep the same sign shape.")
    print()
    print("Press ENTER when ready.")

    input()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        break

    count = 0

    while count < SAMPLES_PER_SIGN:

        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        results = detector.detect(frame)

        frame = detector.draw_landmarks(
            frame,
            results
        )

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            features = extractor.extract(
                hand_landmarks
            )

            writer.writerow(
                list(features) + [label]
            )

            file.flush()

            count += 1

            cv2.putText(
                frame,
                f"Validation {label}: "
                f"{count}/{SAMPLES_PER_SIGN}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
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

        cv2.imshow(
            "V2 Validation Data",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print(
        f"Collected {count} validation samples "
        f"for {label}."
    )


file.close()

print()
print("========================================")
print("V2 VALIDATION COLLECTION COMPLETE")
print("========================================")
print("Saved to:")
print(OUTPUT_FILE)