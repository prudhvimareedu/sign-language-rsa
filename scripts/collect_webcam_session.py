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


CLASSES = [
    "A",
    "B",
    "C",
    "L",
    "O",
    "V",
    "Y"
]

TRAIN_SAMPLES = 20
TEST_SAMPLES = 10

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "webcam_session_train.csv"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "webcam_session_test.csv"
)


def create_writer(file_path):

    file = open(
        file_path,
        "w",
        newline=""
    )

    writer = csv.writer(file)

    header = [
        f"f{i}"
        for i in range(63)
    ]

    header.append("label")

    writer.writerow(header)

    return file, writer


def countdown(label):

    print()
    print(
        f"Get ready for sign: {label}"
    )

    for number in [3, 2, 1]:

        print(number)

        time.sleep(1)


def capture_samples(
    camera,
    detector,
    extractor,
    writer,
    label,
    sample_count,
    mode
):

    collected = 0

    print()
    print(
        f"{mode}: {label}"
    )

    print(
        f"Need {sample_count} samples."
    )

    print(
        "Press SPACE once for each sample."
    )

    print(
        "Change the hand position slightly "
        "between samples."
    )

    while collected < sample_count:

        success, frame = camera.read()

        if not success:
            continue

        results = detector.detect(frame)

        display = frame.copy()

        display = detector.draw_landmarks(
            display,
            results
        )

        cv2.rectangle(
            display,
            (10, 10),
            (680, 145),
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
            f"{mode} samples: "
            f"{collected}/{sample_count}",
            (25, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "SPACE = capture",
            (25, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "ASL Session Collection",
            display
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q"):

            return False

        if key == 32:

            if not results.multi_hand_landmarks:

                print(
                    "No hand detected. "
                    "Move your hand clearly into view."
                )

                continue

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

            print(
                f"{mode} | "
                f"{label} | "
                f"{collected}/{sample_count}"
            )

            # Prevent accidental double capture.
            time.sleep(0.25)

    return True


def main():

    print("========================================")
    print("PROPER WEBCAM SESSION DATASET")
    print("========================================")

    print()
    print(
        "Training samples per class:",
        TRAIN_SAMPLES
    )

    print(
        "Test samples per class:",
        TEST_SAMPLES
    )

    print(
        "Total samples:",
        len(CLASSES)
        * (TRAIN_SAMPLES + TEST_SAMPLES)
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Press SPACE manually for each sample."
    )

    print(
        "Do NOT rapidly press SPACE."
    )

    print(
        "Change your hand position slightly "
        "between captures."
    )

    print(
        "The test samples will NOT be used "
        "during training."
    )

    os.makedirs(
        os.path.dirname(TRAIN_FILE),
        exist_ok=True
    )

    train_file, train_writer = create_writer(
        TRAIN_FILE
    )

    test_file, test_writer = create_writer(
        TEST_FILE
    )

    detector = HandDetector()
    extractor = FeatureExtractor()

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        print()
        print(
            "ERROR: Could not open webcam."
        )

        train_file.close()
        test_file.close()

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    try:

        for label in CLASSES:

            # -------------------------------
            # Training samples
            # -------------------------------

            countdown(label)

            completed = capture_samples(
                camera,
                detector,
                extractor,
                train_writer,
                label,
                TRAIN_SAMPLES,
                "TRAIN"
            )

            if not completed:
                print(
                    "Collection cancelled."
                )
                return

            print()
            print(
                f"Training collection finished "
                f"for {label}."
            )

            time.sleep(1)

            # -------------------------------
            # Test samples
            # -------------------------------

            print()
            print(
                f"Now collect TEST samples "
                f"for {label}."
            )

            print(
                "Change your hand position "
                "slightly before each capture."
            )

            time.sleep(2)

            completed = capture_samples(
                camera,
                detector,
                extractor,
                test_writer,
                label,
                TEST_SAMPLES,
                "TEST"
            )

            if not completed:
                print(
                    "Collection cancelled."
                )
                return

            print()
            print(
                f"Test collection finished "
                f"for {label}."
            )

            time.sleep(1)

    finally:

        train_file.close()
        test_file.close()

        camera.release()
        cv2.destroyAllWindows()

    print()
    print("========================================")
    print("SESSION COLLECTION COMPLETE")
    print("========================================")

    print()
    print("Training file:")
    print(TRAIN_FILE)

    print()
    print("Test file:")
    print(TEST_FILE)

    print()
    print(
        "Training samples:",
        len(CLASSES) * TRAIN_SAMPLES
    )

    print(
        "Test samples:",
        len(CLASSES) * TEST_SAMPLES
    )


if __name__ == "__main__":
    main()