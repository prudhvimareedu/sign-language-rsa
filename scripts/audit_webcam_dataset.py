import os
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


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


def main():

    print("========================================")
    print("WEBCAM DATASET AUDIT")
    print("========================================")

    df = pd.read_csv(DATA_FILE)

    X = df.drop(
        "label",
        axis=1
    ).to_numpy(dtype=np.float32)

    y = df["label"].to_numpy()

    print()
    print("Samples:", len(X))
    print("Features:", X.shape[1])

    print()
    print("Samples per class:")

    counts = pd.Series(y).value_counts().sort_index()

    for label, count in counts.items():
        print(f"{label}: {count}")

    # ----------------------------------------
    # Per-feature variance
    # ----------------------------------------

    variances = np.var(
        X,
        axis=0
    )

    zero_variance = np.sum(
        variances < 1e-8
    )

    print()
    print(
        "Near-zero variance features:",
        zero_variance
    )

    print(
        "Mean feature variance:",
        float(np.mean(variances))
    )

    # ----------------------------------------
    # Class centroids
    # ----------------------------------------

    labels = sorted(
        np.unique(y)
    )

    centroids = {}

    for label in labels:

        class_data = X[y == label]

        centroids[label] = np.mean(
            class_data,
            axis=0
        )

    print()
    print("Average centroid distances:")

    distances = []

    for i in range(len(labels)):

        for j in range(i + 1, len(labels)):

            a = centroids[labels[i]]
            b = centroids[labels[j]]

            distance = np.linalg.norm(
                a - b
            )

            distances.append(distance)

            print(
                f"{labels[i]} vs {labels[j]}: "
                f"{distance:.4f}"
            )

    print()
    print(
        "Mean centroid distance:",
        float(np.mean(distances))
    )

    # ----------------------------------------
    # Confusion matrix
    # ----------------------------------------

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    print()
    print("========================================")
    print("CONFUSION MATRIX")
    print("========================================")

    matrix = confusion_matrix(
        y,
        predictions,
        labels=labels
    )

    print(
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels
        )
    )

    print()
    print("========================================")
    print("CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()