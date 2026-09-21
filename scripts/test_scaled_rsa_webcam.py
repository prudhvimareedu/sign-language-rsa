import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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


def evaluate(name, X, y):

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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
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
    print("SCALED RSA WEBCAM TEST")
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

    selected_indices = joblib.load(
        INDICES_FILE
    )

    X_rsa = X[
        :,
        selected_indices
    ]

    print()
    print(
        "Total samples:",
        len(X)
    )

    print(
        "Original features:",
        X.shape[1]
    )

    print(
        "RSA features:",
        X_rsa.shape[1]
    )

    scaled_63 = evaluate(
        "STANDARD-SCALED 63 FEATURES",
        X,
        y
    )

    scaled_35 = evaluate(
        "STANDARD-SCALED 35 RSA FEATURES",
        X_rsa,
        y
    )

    print()
    print("========================================")
    print("FINAL COMPARISON")
    print("========================================")

    print(
        f"63-feature accuracy: "
        f"{scaled_63:.4f}"
    )

    print(
        f"35-feature RSA accuracy: "
        f"{scaled_35:.4f}"
    )

    print(
        f"RSA difference: "
        f"{scaled_35 - scaled_63:+.4f}"
    )


if __name__ == "__main__":
    main()