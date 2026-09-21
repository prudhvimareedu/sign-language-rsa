import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.vision.feature_extractor import FeatureExtractor


TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sign_landmarks.csv"
)

VALIDATION_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "validation.csv"
)


def transform_file(file_path):

    df = pd.read_csv(file_path)

    labels = df["label"].copy()

    feature_values = df.drop(
        "label",
        axis=1
    ).to_numpy(
        dtype=np.float32
    )

    transformed = []

    for row in feature_values:

        landmarks = row.reshape(
            21,
            3
        )

        normalized = (
            FeatureExtractor.normalize_array(
                landmarks
            )
        )

        transformed.append(
            normalized
        )

    transformed = np.asarray(
        transformed,
        dtype=np.float32
    )

    feature_names = []

    for i in range(21):
        feature_names.append(f"x_{i}")
        feature_names.append(f"y_{i}")
        feature_names.append(f"z_{i}")

    new_df = pd.DataFrame(
        transformed,
        columns=feature_names
    )

    new_df["label"] = labels.values

    new_df.to_csv(
        file_path,
        index=False
    )

    print(
        os.path.basename(file_path),
        "updated:",
        len(new_df),
        "samples"
    )


def main():

    print("========================================")
    print("REBUILDING FEATURE REPRESENTATION")
    print("========================================")

    transform_file(TRAIN_FILE)
    transform_file(VALIDATION_FILE)

    print()
    print("Both datasets now use the same")
    print("canonical hand representation.")
    print("========================================")


if __name__ == "__main__":
    main()