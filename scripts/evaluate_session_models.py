import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "webcam_session_train.csv"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "webcam_session_test.csv"
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


def build_model():
    return Pipeline(
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


def main():

    print("========================================")
    print("SESSION MODEL EVALUATION")
    print("========================================")

    # -------------------------------------
    # Load train/test session datasets
    # -------------------------------------

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    X_train = train_df.drop(
        "label",
        axis=1
    ).to_numpy(
        dtype=np.float32
    )

    y_train = train_df[
        "label"
    ].to_numpy()

    X_test = test_df.drop(
        "label",
        axis=1
    ).to_numpy(
        dtype=np.float32
    )

    y_test = test_df[
        "label"
    ].to_numpy()

    print()
    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    print(
        "Original features:",
        X_train.shape[1]
    )

    # -------------------------------------
    # FULL 63-FEATURE MODEL
    # -------------------------------------

    print()
    print("----------------------------------------")
    print("STANDARD-SCALED 63 FEATURES")
    print("----------------------------------------")

    full_model = build_model()

    full_model.fit(
        X_train,
        y_train
    )

    full_predictions = full_model.predict(
        X_test
    )

    full_accuracy = accuracy_score(
        y_test,
        full_predictions
    )

    print(
        f"Accuracy: {full_accuracy:.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            full_predictions,
            zero_division=0
        )
    )

    # -------------------------------------
    # Save FULL 63-feature model
    # -------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    full_model_path = os.path.join(
        MODEL_DIR,
        "session_baseline_scaled_svm.joblib"
    )

    joblib.dump(
        full_model,
        full_model_path
    )

    print()
    print(
        "63-feature model saved to:"
    )

    print(
        full_model_path
    )

    # -------------------------------------
    # RSA MODEL
    # -------------------------------------

    selected_indices = joblib.load(
        INDICES_FILE
    )

    X_train_rsa = X_train[
        :,
        selected_indices
    ]

    X_test_rsa = X_test[
        :,
        selected_indices
    ]

    print()
    print(
        "RSA selected features:",
        len(selected_indices)
    )

    print()
    print("----------------------------------------")
    print("STANDARD-SCALED 35 RSA FEATURES")
    print("----------------------------------------")

    rsa_model = build_model()

    rsa_model.fit(
        X_train_rsa,
        y_train
    )

    rsa_predictions = rsa_model.predict(
        X_test_rsa
    )

    rsa_accuracy = accuracy_score(
        y_test,
        rsa_predictions
    )

    print(
        f"Accuracy: {rsa_accuracy:.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            rsa_predictions,
            zero_division=0
        )
    )

    # -------------------------------------
    # Save RSA model
    # -------------------------------------

    rsa_model_path = os.path.join(
        MODEL_DIR,
        "session_final_scaled_rsa_svm.joblib"
    )

    joblib.dump(
        rsa_model,
        rsa_model_path
    )

    print()
    print(
        "RSA model saved to:"
    )

    print(
        rsa_model_path
    )

    # -------------------------------------
    # Final comparison
    # -------------------------------------

    print()
    print("========================================")
    print("FINAL COMPARISON")
    print("========================================")

    print(
        f"63-feature accuracy: "
        f"{full_accuracy:.4f}"
    )

    print(
        f"35-feature RSA accuracy: "
        f"{rsa_accuracy:.4f}"
    )

    print(
        f"Difference: "
        f"{rsa_accuracy - full_accuracy:+.4f}"
    )

    print()
    print("Both models are saved successfully.")


if __name__ == "__main__":
    main()