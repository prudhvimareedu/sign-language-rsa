import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


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
    "webcam_calibration.csv"
)

INDICES_FILE = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "asl_rsa_selected_indices.joblib"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)


def main():

    print("========================================")
    print("FINAL SCALED RSA WEBCAM MODEL")
    print("========================================")

    df = pd.read_csv(DATA_FILE)

    X = df.drop(
        "label",
        axis=1
    ).to_numpy(
        dtype=np.float32
    )

    y = df["label"].to_numpy()

    selected_indices = joblib.load(
        INDICES_FILE
    )

    X_selected = X[
        :,
        selected_indices
    ]

    print()
    print("Total samples:", len(X))
    print("Original features:", X.shape[1])
    print("RSA features:", X_selected.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    class_weight="balanced",
                    cache_size=4096
                )
            )
        ]
    )

    print()
    print("Training final scaled RSA model...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("========================================")
    print("FINAL MODEL RESULTS")
    print("========================================")

    print(
        f"Validation accuracy: {accuracy:.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        "final_scaled_rsa_svm.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    metadata = {
        "original_features": 63,
        "rsa_features": int(
            len(selected_indices)
        ),
        "feature_reduction_percent": 44.4444444444,
        "validation_accuracy": float(
            accuracy
        ),
        "classes": sorted(
            np.unique(y).tolist()
        )
    }

    metadata_path = os.path.join(
        MODEL_DIR,
        "final_scaled_rsa_metadata.joblib"
    )

    joblib.dump(
        metadata,
        metadata_path
    )

    print()
    print("Model saved to:")
    print(model_path)

    print()
    print("Metadata saved to:")
    print(metadata_path)


if __name__ == "__main__":
    main()