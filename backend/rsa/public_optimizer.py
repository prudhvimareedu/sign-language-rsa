import os
import sys
import joblib
import numpy as np

from sklearn.model_selection import train_test_split


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.rsa.public_fitness import evaluate_public_subset
from backend.rsa.reptile_search import (
    ReptileSearchFeatureSelector
)


TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "public_hog_train.npz"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)


def run_public_rsa():

    print("========================================")
    print("PUBLIC DATASET RSA FEATURE SELECTION")
    print("========================================")

    data = np.load(TRAIN_FILE)

    X = data["X"]
    y = data["y"]

    print(
        "Full training data:",
        X.shape
    )

    # -------------------------------------------------
    # Use a representative subset for RSA search
    # -------------------------------------------------

    X_search, _, y_search, _ = train_test_split(
        X,
        y,
        train_size=8000,
        random_state=42,
        stratify=y
    )

    X_internal_train, X_validation, \
        y_internal_train, y_validation = train_test_split(
            X_search,
            y_search,
            test_size=2000,
            random_state=42,
            stratify=y_search
        )

    print(
        "RSA training samples:",
        len(X_internal_train)
    )

    print(
        "RSA validation samples:",
        len(X_validation)
    )

    print(
        "Available features:",
        X.shape[1]
    )

    # -------------------------------------------------
    # Fitness function
    # -------------------------------------------------

    def fitness(mask):

        return evaluate_public_subset(
            mask,
            X_internal_train,
            y_internal_train,
            X_validation,
            y_validation
        )

    # -------------------------------------------------
    # RSA
    # -------------------------------------------------

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

    print()
    print("========================================")
    print("PUBLIC RSA RESULT")
    print("========================================")

    print(
        "Original features:",
        X.shape[1]
    )

    print(
        "Selected features:",
        len(selected_indices)
    )

    print(
        "Feature reduction:",
        f"{(1 - len(selected_indices) / X.shape[1]) * 100:.2f}%"
    )

    print()
    print("Selected feature indices:")
    print(selected_indices)

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    output = {
        "selected_indices": selected_indices,
        "feature_mask": best_mask,
        "original_feature_count": X.shape[1],
        "fitness_history": selector.fitness_history
    }

    output_path = os.path.join(
        MODEL_DIR,
        "public_rsa_features.joblib"
    )

    joblib.dump(
        output,
        output_path
    )

    print()
    print(
        "RSA feature selection saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    run_public_rsa()