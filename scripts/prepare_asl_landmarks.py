import os
import sys
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.vision.hand_detector import HandDetector
from backend.vision.feature_extractor import FeatureExtractor


# =====================================================
# Settings
# =====================================================

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "asl_alphabet",
    "asl_alphabet_train"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "asl_landmarks.csv"
)

# Number of images per class for the first proper model
IMAGES_PER_CLASS = 500

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# =====================================================
# Setup
# =====================================================

detector = HandDetector()
extractor = FeatureExtractor()

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# =====================================================
# Find classes
# =====================================================

classes = sorted(
    [
        name
        for name in os.listdir(DATASET_DIR)
        if os.path.isdir(
            os.path.join(
                DATASET_DIR,
                name
            )
        )
    ]
)

print("========================================")
print("ASL PUBLIC DATASET → HAND LANDMARKS")
print("========================================")

print(
    "Classes found:",
    len(classes)
)

print(classes)

print(
    "Images per class:",
    IMAGES_PER_CLASS
)

print("========================================")


# =====================================================
# Feature names
# =====================================================

feature_names = []

for i in range(21):

    feature_names.append(
        f"x_{i}"
    )

    feature_names.append(
        f"y_{i}"
    )

    feature_names.append(
        f"z_{i}"
    )

feature_names.append(
    "label"
)


# =====================================================
# Process images
# =====================================================

rows = []

total_processed = 0
total_detected = 0
total_failed = 0


for class_name in classes:

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    image_files = []

    for filename in os.listdir(
        class_dir
    ):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in VALID_EXTENSIONS:

            image_files.append(
                filename
            )

    image_files = sorted(
        image_files
    )[:IMAGES_PER_CLASS]

    print()
    print(
        f"Processing {class_name}: "
        f"{len(image_files)} images"
    )

    class_detected = 0

    for index, filename in enumerate(
        image_files,
        start=1
    ):

        image_path = os.path.join(
            class_dir,
            filename
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            total_failed += 1
            continue

        results = detector.detect(
            image
        )

        total_processed += 1

        if not results.multi_hand_landmarks:

            total_failed += 1
            continue

        # Use first detected hand
        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )

        features = extractor.extract(
            hand_landmarks
        )

        rows.append(
            list(features) + [class_name]
        )

        total_detected += 1
        class_detected += 1

        if index % 100 == 0:

            print(
                f"  {index}/{len(image_files)} "
                f"| detected: {class_detected}"
            )


    print(
        f"Finished {class_name} "
        f"| detected: {class_detected}/"
        f"{len(image_files)}"
    )


# =====================================================
# Save dataset
# =====================================================

print()
print("========================================")
print("PROCESSING COMPLETE")
print("========================================")

print(
    "Images processed:",
    total_processed
)

print(
    "Hands detected:",
    total_detected
)

print(
    "Detection failures:",
    total_failed
)


if not rows:

    print(
        "ERROR: No hand landmarks were extracted."
    )

    sys.exit(1)


df = pd.DataFrame(
    rows,
    columns=feature_names
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(
    "Final dataset shape:",
    df.shape
)

print()
print("Class counts:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print()
print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)

print()
print("========================================")
print("ASL LANDMARK DATASET READY")
print("========================================")