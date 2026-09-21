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


PUBLIC_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "asl_landmarks.csv"
)

WEBCAM_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "webcam_calibration.csv"
)

RSA_INDICES_FILE = os.path.join(
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
    print("WEBCAM-ADAPTED RSA MODEL")
    print("========================================")

    # -------------------------------------
    # Load public dataset
    # -------------------------------------

    public_df = pd.read_csv(
        PUBLIC_FILE
    )

    X_public = public_df.drop(
        "label",
        axis=1
    ).to_numpy(
        dtype=np.float32
    )

    y_public = public_df["label"].to_numpy()

    print(
        "Public samples:",
        len(X_public)
    )

    # -------------------------------------
    # Load webcam calibration data
    # -------------------------------------

    webcam_df = pd.read_csv(
        WEBCAM_FILE
    )

    X_webcam = webcam_df.drop(
        "label",
        axis=1
    ).to_numpy(
        dtype=np.float32
    )

    y_webcam = webcam_df["label"].to_numpy()

    print(
        "Webcam samples:",
        len(X_webcam)
    )

    # -------------------------------------
    # Split public data
    # -------------------------------------

    X_train_public, X_test_public, y_train_public, y_test_public = (
        train_test_split(
            X_public,
            y_public,
            test_size=0.20,
            random_state=42,
            stratify=y_public
        )
    )

    # -------------------------------------
    # Split webcam calibration data
    # -------------------------------------

    X_train_webcam, X_test_webcam, y_train_webcam, y_test_webcam = (
        train_test_split(
            X_webcam,
            y_webcam,
            test_size=0.20,
            random_state=42,
            stratify=y_webcam
        )
    )

    print()
    print(
        "Public training:",
        len(X_train_public)
    )

    print(
        "Public testing:",
        len(X_test_public)
    )

    print(
        "Webcam training:",
        len(X_train_webcam)
    )

    print(
        "Webcam testing:",
        len(X_test_webcam)
    )

    # -------------------------------------
    # Combine training data
    # -------------------------------------

    X_train = np.vstack(
        [
            X_train_public,
            X_train_webcam
        ]
    )

    y_train = np.concatenate(
        [
            y_train_public,
            y_train_webcam
        ]
    )

    # -------------------------------------
    # Load RSA-selected feature indices
    # -------------------------------------

    selected_indices = joblib.load(
        RSA_INDICES_FILE
    )

    print()
    print(
        "RSA selected features:",
        len(selected_indices)
    )

    X_train_selected = X_train[
        :,
        selected_indices
    ]

    X_public_test_selected = X_test_public[
        :,
        selected_indices
    ]

    X_webcam_test_selected = X_test_webcam[
        :,
        selected_indices
    ]

    # -------------------------------------
    # Sample weights
    # -------------------------------------
    #
    # Public data gets normal weight.
    # Webcam samples get stronger influence.
    #

    public_weight = np.ones(
        len(X_train_public),
        dtype=np.float32
    )

    webcam_weight = np.full(
        len(X_train_webcam),
        15.0,
        dtype=np.float32
    )

    sample_weights = np.concatenate(
        [
            public_weight,
            webcam_weight
        ]
    )

    # -------------------------------------
    # Train adapted RSA SVM
    # -------------------------------------

    print()
    print(
        "Training webcam-adapted RSA RBF-SVM..."
    )

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        cache_size=4096
    )

    model.fit(
        X_train_selected,
        y_train,
        sample_weight=sample_weights
    )

    print(
        "Training completed."
    )

    # -------------------------------------
    # Public test evaluation
    # -------------------------------------

    public_predictions = model.predict(
        X_public_test_selected
    )

    public_accuracy = accuracy_score(
        y_test_public,
        public_predictions
    )

    # -------------------------------------
    # Webcam test evaluation
    # -------------------------------------

    webcam_predictions = model.predict(
        X_webcam_test_selected
    )

    webcam_accuracy = accuracy_score(
        y_test_webcam,
        webcam_predictions
    )

    print()
    print("========================================")
    print("ADAPTED MODEL RESULTS")
    print("========================================")

    print(
        f"Public test accuracy: "
        f"{public_accuracy:.4f}"
    )

    print(
        f"Webcam validation accuracy: "
        f"{webcam_accuracy:.4f}"
    )

    print()
    print("Webcam validation report:")
    print(
        classification_report(
            y_test_webcam,
            webcam_predictions,
            zero_division=0
        )
    )

    # -------------------------------------
    # Save model
    # -------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        "asl_rsa_webcam_adapted_svm.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    metrics = {
        "public_accuracy": float(
            public_accuracy
        ),
        "webcam_validation_accuracy": float(
            webcam_accuracy
        ),
        "rsa_features": int(
            len(selected_indices)
        ),
        "webcam_training_samples": int(
            len(X_train_webcam)
        ),
        "webcam_weight": 15.0
    }

    metrics_path = os.path.join(
        MODEL_DIR,
        "asl_rsa_webcam_adapted_metrics.joblib"
    )

    joblib.dump(
        metrics,
        metrics_path
    )

    print()
    print(
        "Adapted model saved to:"
    )

    print(model_path)

    print()
    print(
        "Metrics saved to:"
    )

    print(metrics_path)


if __name__ == "__main__":
    main()