import os
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PUBLIC_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "public"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

TRAIN_FILE = os.path.join(
    PUBLIC_DIR,
    "sign_mnist_train.csv"
)

TEST_FILE = os.path.join(
    PUBLIC_DIR,
    "sign_mnist_test.csv"
)

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "public_hog_train.npz"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "public_hog_test.npz"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def create_hog():

    return cv2.HOGDescriptor(
        (28, 28),   # window size
        (14, 14),   # block size
        (7, 7),     # block stride
        (7, 7),     # cell size
        9           # orientations
    )


def extract_features(df, hog):

    labels = df["label"].to_numpy(
        dtype=np.int64
    )

    pixels = df.drop(
        "label",
        axis=1
    ).to_numpy(
        dtype=np.uint8
    )

    features = []

    total = len(pixels)

    for index, row in enumerate(pixels):

        image = row.reshape(
            28,
            28
        )

        descriptor = hog.compute(
            image
        )

        features.append(
            descriptor.flatten()
        )

        if (index + 1) % 5000 == 0:

            print(
                f"Processed "
                f"{index + 1}/{total} images"
            )

    return (
        np.asarray(
            features,
            dtype=np.float32
        ),
        labels
    )


def process_file(
    input_file,
    output_file,
    name
):

    print()
    print(
        f"Loading {name} dataset..."
    )

    df = pd.read_csv(
        input_file
    )

    print(
        f"{name} samples:",
        len(df)
    )

    hog = create_hog()

    X, y = extract_features(
        df,
        hog
    )

    print(
        f"{name} HOG shape:",
        X.shape
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    print(
        f"Saved: {output_file}"
    )


def main():

    print("========================================")
    print("PUBLIC DATASET HOG PREPARATION")
    print("========================================")

    process_file(
        TRAIN_FILE,
        TRAIN_OUTPUT,
        "training"
    )

    process_file(
        TEST_FILE,
        TEST_OUTPUT,
        "testing"
    )

    print()
    print("========================================")
    print("HOG PREPARATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()