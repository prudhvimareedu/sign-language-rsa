import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


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

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)


def main():

    print("========================================")
    print("VALIDATION PIPELINE DIAGNOSTIC")
    print("========================================")

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    X_train = train_df.drop(
        "label",
        axis=1
    )

    y_train = train_df["label"]

    X_validation = validation_df.drop(
        "label",
        axis=1
    )

    y_validation = validation_df["label"]

    print()
    print("Training shape:", X_train.shape)
    print("Validation shape:", X_validation.shape)

    print()
    print("Training feature range:")
    print(
        "min =",
        X_train.min().min()
    )
    print(
        "max =",
        X_train.max().max()
    )

    print()
    print("Validation feature range:")
    print(
        "min =",
        X_validation.min().min()
    )
    print(
        "max =",
        X_validation.max().max()
    )

    print()
    print("NaN values:")
    print(
        "Training:",
        X_train.isna().sum().sum()
    )
    print(
        "Validation:",
        X_validation.isna().sum().sum()
    )

    # Load current scaler and baseline
    scaler = joblib.load(
        os.path.join(
            MODELS_DIR,
            "feature_scaler.joblib"
        )
    )

    model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "baseline_model.joblib"
        )
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    predictions = model.predict(
        X_validation_scaled
    )

    print()
    print("Actual validation labels:")
    print(
        y_validation.value_counts().sort_index()
    )

    print()
    print("Predicted validation labels:")
    print(
        pd.Series(predictions).value_counts().sort_index()
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    print()
    print(
        f"Validation accuracy: {accuracy:.4f}"
    )

    print()
    print("First 10 predictions:")
    print(predictions[:10])


if __name__ == "__main__":
    main()