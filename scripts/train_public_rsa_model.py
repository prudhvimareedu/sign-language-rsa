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


def main():

    print("========================================")
    print("RSA + SVM PUBLIC MODEL")
    print("========================================")

    # -----------------------------------------
    # Load public HOG data
    # -----------------------------------------

    train_data = np.load(TRAIN_FILE)
    test_data = np.load(TEST_FILE)

    X_train = train_data["X"]
    y_train = train_data["y"]

    X_test = test_data["X"]
    y_test = test_data["y"]

    # -----------------------------------------
    # Load RSA feature selection
    # -----------------------------------------

    rsa_path = os.path.join(
        MODEL_DIR,
        "public_rsa_features.joblib"
    )

    rsa_data = joblib.load(
        rsa_path
    )

    selected_indices = np.array(
        rsa_data["selected_indices"]
    )

    print()
    print(
        "Original features:",
        X_train.shape[1]
    )

    print(
        "RSA selected features:",
        len(selected_indices)
    )

    # -----------------------------------------
    # Select RSA features
    # -----------------------------------------

    X_train_selected = X_train[
        :,
        selected_indices
    ]

    X_test_selected = X_test[
        :,
        selected_indices
    ]

    # -----------------------------------------
    # Train final RBF-SVM
    # -----------------------------------------

    print()
    print(
        "Training RBF-SVM with RSA features..."
    )

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        cache_size=4096
    )

    model.fit(
        X_train_selected,
        y_train
    )

    print(
        "Training completed."
    )

    # -----------------------------------------
    # Official test evaluation
    # -----------------------------------------

    print()
    print(
        "Evaluating on official test set..."
    )

    predictions = model.predict(
        X_test_selected
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("========================================")
    print("RSA + SVM RESULTS")
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
    # Save model
    # -----------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "public_rsa_svm.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print()
    print(
        "RSA + SVM model saved to:"
    )

    print(
        model_path
    )

    # Save selected indices separately for inference
    indices_path = os.path.join(
        MODEL_DIR,
        "public_rsa_selected_indices.joblib"
    )

    joblib.dump(
        selected_indices,
        indices_path
    )

    print(
        "Selected indices saved to:"
    )

    print(
        indices_path
    )


if __name__ == "__main__":
    main()