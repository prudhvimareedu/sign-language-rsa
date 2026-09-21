import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


DATA_FILE = os.path.join(
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
    print("FRESH VALIDATION TEST")
    print("========================================")

    # Load fresh validation data
    df = pd.read_csv(DATA_FILE)

    X = df.drop(
        "label",
        axis=1
    )

    y = df["label"]

    print(
        "Validation samples:",
        len(df)
    )

    print()
    print("Validation class counts:")
    print(
        y.value_counts().sort_index()
    )

    # Load scaler
    scaler = joblib.load(
        os.path.join(
            MODELS_DIR,
            "feature_scaler.joblib"
        )
    )

    # Load RSA feature selection
    rsa_info = joblib.load(
        os.path.join(
            MODELS_DIR,
            "rsa_feature_selection.joblib"
        )
    )

    selected_indices = np.array(
        rsa_info["selected_indices"]
    )

    # Load RSA model
    rsa_model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "rsa_model.joblib"
        )
    )

    # Scale fresh data
    X_scaled = scaler.transform(
        X
    )

    # Select RSA features
    X_selected = X_scaled[
        :,
        selected_indices
    ]

    # Predict
    predictions = rsa_model.predict(
        X_selected
    )

    # Metrics
    accuracy = accuracy_score(
        y,
        predictions
    )

    print()
    print("========================================")
    print("FRESH VALIDATION RESULTS")
    print("========================================")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y,
            predictions
        )
    )


if __name__ == "__main__":
    main()