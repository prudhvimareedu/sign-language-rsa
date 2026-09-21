import os
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "public_hog_train.npz"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "public_hog_test.npz"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


def main():

    print("========================================")
    print("PUBLIC DATASET BASELINE SVM")
    print("========================================")

    # Load HOG data
    train_data = np.load(TRAIN_FILE)
    test_data = np.load(TEST_FILE)

    X_train = train_data["X"]
    y_train = train_data["y"]

    X_test = test_data["X"]
    y_test = test_data["y"]

    print()
    print("Training shape:", X_train.shape)
    print("Testing shape :", X_test.shape)
    print("Features      :", X_train.shape[1])

    # -----------------------------------------
    # Train SVM
    # -----------------------------------------

    print()
    print("Training SVM...")

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        cache_size=4096
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------

    print()
    print("Evaluating...")

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("========================================")
    print("PUBLIC BASELINE RESULTS")
    print("========================================")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # -----------------------------------------
    # Save
    # -----------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "public_baseline_svm.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Model saved to:"
    )

    print(
        model_path
    )


if __name__ == "__main__":
    main()