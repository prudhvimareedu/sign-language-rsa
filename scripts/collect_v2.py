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
    "sign_landmarks.csv"
)

SAMPLES_PER_SIGN = 100

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


detector = HandDetector()
extractor = FeatureExtractor()


# Create a fresh dataset
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
print("CLEAN V2 SIGN DATASET COLLECTOR")
print("========================================")
print("For testing, collect only:")
print("A, B, C")
print("100 samples per sign")
print("Type EXIT to stop")
print("========================================")


while True:

    label = input(
        "\nEnter sign label: "
    ).strip().upper()

    if label == "EXIT":
        break

    if label not in ["A", "B", "C"]:

        print(
            "Enter only A, B, or C for this test."
        )
        continue

    print()
    print(f"Prepare sign: {label}")
    print()
    print("When ready, press ENTER.")
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

        frame = cv2.flip(
            frame,
            1
        )

        results = detector.detect(
            frame
        )

        frame = detector.draw_landmarks(
            frame,
            results
        )

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            # IMPORTANT:
            # FeatureExtractor performs the
            # normalization exactly once.
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
                f"V2 {label}: {count}/{SAMPLES_PER_SIGN}",
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
            "Clean V2 Dataset",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print(
        f"Collected {count} samples for {label}."
    )


file.close()

print()
print("========================================")
print("V2 COLLECTION COMPLETE")
print("========================================")
print(
    "Saved to:",
    OUTPUT_FILE
)