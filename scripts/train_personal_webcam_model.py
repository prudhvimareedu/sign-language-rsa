import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
    print("PERSONAL WEBCAM RSA MODEL")
    print("========================================")

    df = pd.read_csv(
        DATA_FILE
    )

    X = df.drop(
        "label",
        axis=1
    ).to_numpy(
        dtype=np.float32
    )

    y = df["label"].to_numpy()

    print(
        "Total samples:",
        len(X)
    )

    print(
        "Classes:",
        sorted(set(y))
    )

    selected_indices = joblib.load(
        INDICES_FILE
    )

    print(
        "RSA features:",
        len(selected_indices)
    )

    X = X[
        :,
        selected_indices
    ]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print()
    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_test)
    )

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        cache_size=4096
    )

    print()
    print(
        "Training personal webcam model..."
    )

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
    print("PERSONAL MODEL RESULTS")
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
        "personal_webcam_rsa_svm.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print()
    print(
        "Model saved to:"
    )

    print(model_path)


if __name__ == "__main__":
    main()