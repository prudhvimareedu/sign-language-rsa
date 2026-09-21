import os
import sys
import csv
import cv2

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.vision.hand_detector import HandDetector
from backend.vision.feature_extractor import FeatureExtractor


# ==============================
# Settings
# ==============================

SAMPLES_PER_SIGN = 100

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sign_landmarks.csv"
)


# ==============================
# Create output folder
# ==============================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ==============================
# Initialize
# ==============================

detector = HandDetector()
extractor = FeatureExtractor()


# ==============================
# CSV setup
# ==============================

file_exists = os.path.exists(OUTPUT_FILE)

file = open(
    OUTPUT_FILE,
    "a",
    newline=""
)

writer = csv.writer(file)

if not file_exists:

    header = []

    for i in range(21):
        header.append(f"x_{i}")
        header.append(f"y_{i}")
        header.append(f"z_{i}")

    header.append("label")

    writer.writerow(header)


# ==============================
# Main collection loop
# ==============================

print()
print("======================================")
print("SIGN LANGUAGE DATASET COLLECTOR")
print("======================================")
print("Enter a letter A-Y.")
print("J and Z will be handled later.")
print("Enter EXIT to quit.")
print("======================================")


while True:

    label = input("\nEnter sign label: ").strip().upper()

    if label == "EXIT":
        break

    if len(label) != 1 or not label.isalpha():

        print("Please enter one letter only.")
        continue

    if label in ["J", "Z"]:

        print(
            "J and Z involve movement. "
            "We will handle them later."
        )
        continue


    print()
    print(f"Collecting sign: {label}")
    print(f"Target samples: {SAMPLES_PER_SIGN}")
    print("Show the sign clearly.")
    print("Move your hand slightly between samples.")
    print("Press Q to stop this collection.")
    print()


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

            hand_landmarks = results.multi_hand_landmarks[0]

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
                f"Sign {label}: {count}/{SAMPLES_PER_SIGN}",
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
            "Sign Language Dataset Collector",
            frame
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


    camera.release()
    cv2.destroyAllWindows()


    print()
    print(f"Finished sign {label}.")
    print(f"Samples collected: {count}")
    print()


file.close()

print()
print("======================================")
print("DATA COLLECTION STOPPED")
print("Dataset saved to:")
print(OUTPUT_FILE)
print("======================================")