import os
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sign_landmarks.csv"
)


def normalize_features(row):

    # First 63 values are x,y,z coordinates
    values = row.iloc[:63].to_numpy(
        dtype=np.float32
    )

    # Convert 63 values → 21 × 3
    landmarks = values.reshape(
        21,
        3
    )

    # -----------------------------------------
    # 1. Wrist becomes origin
    # -----------------------------------------

    wrist = landmarks[0].copy()

    landmarks = landmarks - wrist

    # -----------------------------------------
    # 2. Normalize hand size
    # -----------------------------------------

    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    scale = np.max(distances)

    if scale < 1e-6:
        scale = 1.0

    landmarks = landmarks / scale

    # -----------------------------------------
    # 3. Flatten back to 63 features
    # -----------------------------------------

    return landmarks.flatten()


def main():

    print("========================================")
    print("NORMALIZING EXISTING SIGN DATASET")
    print("========================================")

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "Original samples:",
        len(df)
    )

    print(
        "Original features:",
        len(df.columns) - 1
    )

    feature_columns = [
        column
        for column in df.columns
        if column != "label"
    ]

    normalized_features = []

    for _, row in df.iterrows():

        normalized_features.append(
            normalize_features(row)
        )

    normalized_array = np.array(
        normalized_features,
        dtype=np.float32
    )

    normalized_df = pd.DataFrame(
        normalized_array,
        columns=feature_columns
    )

    normalized_df["label"] = (
        df["label"].values
    )

    # Replace the old dataset
    normalized_df.to_csv(
        DATA_FILE,
        index=False
    )

    print()
    print(
        "Normalized samples:",
        len(normalized_df)
    )

    print(
        "Normalized features:",
        len(normalized_df.columns) - 1
    )

    print()
    print(
        "Dataset updated successfully:"
    )

    print(DATA_FILE)


if __name__ == "__main__":
    main()