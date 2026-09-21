import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


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


def evaluate_model(X, y, name):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        cache_size=4096
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print()
    print("----------------------------------------")
    print(name)
    print("----------------------------------------")

    print(
        "Fold accuracies:",
        np.round(scores, 4)
    )

    print(
        f"Mean accuracy: {scores.mean():.4f}"
    )

    print(
        f"Std deviation: {scores.std():.4f}"
    )

    return scores.mean()


def main():

    print("========================================")
    print("WEBCAM 63 vs RSA 35 FEATURES")
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

    print()
    print("Samples:", len(X))
    print("Original features:", X.shape[1])

    selected_indices = joblib.load(
        INDICES_FILE
    )

    X_rsa = X[
        :,
        selected_indices
    ]

    print(
        "RSA features:",
        X_rsa.shape[1]
    )

    accuracy_63 = evaluate_model(
        X,
        y,
        "ALL 63 FEATURES"
    )

    accuracy_35 = evaluate_model(
        X_rsa,
        y,
        "RSA 35 FEATURES"
    )

    print()
    print("========================================")
    print("FINAL COMPARISON")
    print("========================================")

    print(
        f"63-feature accuracy: "
        f"{accuracy_63:.4f}"
    )

    print(
        f"35-feature accuracy: "
        f"{accuracy_35:.4f}"
    )

    print(
        f"Difference: "
        f"{accuracy_63 - accuracy_35:+.4f}"
    )


if __name__ == "__main__":
    main()