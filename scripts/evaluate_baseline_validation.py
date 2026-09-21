import os
import sys
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report
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
    print("BASELINE FRESH VALIDATION TEST")
    print("========================================")

    df = pd.read_csv(DATA_FILE)

    X = df.drop(
        "label",
        axis=1
    )

    y = df["label"]

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

    X_scaled = scaler.transform(X)

    predictions = model.predict(
        X_scaled
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    print()
    print("Fresh samples:", len(df))
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


if __name__ == "__main__":
    main()