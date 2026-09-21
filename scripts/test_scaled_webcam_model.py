import os
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


def evaluate(name, model, X, y):

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
    print("SCALED WEBCAM MODEL TEST")
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
        "Samples:",
        len(X)
    )

    print(
        "Features:",
        X.shape[1]
    )

    # Existing approach
    raw_model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced"
    )

    # Scaled approach
    scaled_model = Pipeline(
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
                    class_weight="balanced"
                )
            )
        ]
    )

    raw_accuracy = evaluate(
        "RAW 63 FEATURES",
        raw_model,
        X,
        y
    )

    scaled_accuracy = evaluate(
        "STANDARD-SCALED 63 FEATURES",
        scaled_model,
        X,
        y
    )

    print()
    print("========================================")
    print("FINAL RESULT")
    print("========================================")

    print(
        f"Raw accuracy:    {raw_accuracy:.4f}"
    )

    print(
        f"Scaled accuracy: {scaled_accuracy:.4f}"
    )

    print(
        f"Improvement:     "
        f"{scaled_accuracy - raw_accuracy:+.4f}"
    )


if __name__ == "__main__":
    main()