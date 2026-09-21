import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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

sys.path.insert(0, PROJECT_ROOT)

from backend.rsa.reptile_search import (
    ReptileSearchFeatureSelector
)

from backend.rsa.public_fitness import (
    evaluate_public_subset
)


DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "asl_landmarks.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)


def main():

    print("========================================")
    print("ASL LANDMARK RSA OPTIMIZATION")
    print("========================================")

    # -----------------------------------------
    # Load data
    # -----------------------------------------

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
        "Original features:",
        X.shape[1]
    )

    # -----------------------------------------
    # Same 80/20 split as baseline
    # -----------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # -----------------------------------------
    # Internal RSA validation split
    # -----------------------------------------

    (
        X_rsa_train,
        X_rsa_validation,
        y_rsa_train,
        y_rsa_validation
    ) = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    print()
    print(
        "RSA training samples:",
        len(X_rsa_train)
    )

    print(
        "RSA validation samples:",
        len(X_rsa_validation)
    )

    # -----------------------------------------
    # RSA fitness
    # -----------------------------------------

    def fitness(mask):

        return evaluate_public_subset(
            mask,
            X_rsa_train,
            y_rsa_train,
            X_rsa_validation,
            y_rsa_validation
        )

    # -----------------------------------------
    # Run RSA
    # -----------------------------------------

    selector = ReptileSearchFeatureSelector(
        population_size=10,
        iterations=10,
        random_state=42
    )

    best_mask = selector.optimize(
        fitness_function=fitness,
        number_of_features=X.shape[1]
    )

    selected_indices = np.where(
        best_mask == 1
    )[0]

    # -----------------------------------------
    # Ensure enough features
    # -----------------------------------------

    if len(selected_indices) < 5:

        print(
            "RSA selected too few features."
        )

        sys.exit(1)

    print()
    print("========================================")
    print("ASL RSA RESULT")
    print("========================================")

    print(
        "Original features:",
        X.shape[1]
    )

    print(
        "Selected features:",
        len(selected_indices)
    )

    reduction = (
        1
        - len(selected_indices)
        / X.shape[1]
    ) * 100

    print(
        f"Feature reduction: {reduction:.2f}%"
    )

    print()
    print(
        "Selected feature indices:"
    )

    print(
        selected_indices
    )

    # -----------------------------------------
    # Select features
    # -----------------------------------------

    X_train_selected = (
        X_train[
            :,
            selected_indices
        ]
    )

    X_test_selected = (
        X_test[
            :,
            selected_indices
        ]
    )

    # -----------------------------------------
    # Final RBF-SVM
    # -----------------------------------------

    print()
    print(
        "Training RSA-selected RBF-SVM..."
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
        y_train
    )

    print(
        "Training completed."
    )

    # -----------------------------------------
    # Final untouched test set
    # -----------------------------------------

    predictions = model.predict(
        X_test_selected
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("========================================")
    print("ASL RSA MODEL RESULTS")
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

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        "asl_rsa_svm.joblib"
    )

    indices_path = os.path.join(
        MODEL_DIR,
        "asl_rsa_selected_indices.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        selected_indices,
        indices_path
    )

    print()
    print(
        "RSA model saved to:"
    )

    print(
        model_path
    )

    print()
    print(
        "Selected indices saved to:"
    )

    print(
        indices_path
    )

    # -----------------------------------------
    # Save comparison
    # -----------------------------------------

    comparison = {
        "baseline_features": int(X.shape[1]),
        "baseline_accuracy": 0.9806,
        "rsa_features": int(len(selected_indices)),
        "rsa_accuracy": float(accuracy),
        "feature_reduction_percent": float(
            reduction
        )
    }

    comparison_path = os.path.join(
        MODEL_DIR,
        "asl_model_comparison.joblib"
    )

    joblib.dump(
        comparison,
        comparison_path
    )

    print()
    print(
        "Comparison saved to:"
    )

    print(
        comparison_path
    )


if __name__ == "__main__":
    main()