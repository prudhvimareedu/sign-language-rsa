import os
import sys
import joblib
import numpy as np

from sklearn.model_selection import train_test_split


# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, PROJECT_ROOT)


from backend.ai.preprocessing import prepare_data
from backend.rsa.fitness import evaluate_feature_subset
from backend.rsa.reptile_search import ReptileSearchFeatureSelector


def run_rsa():

    print("========================================")
    print("RSA FEATURE SELECTION")
    print("========================================")

    # Load the original dataset split
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # -----------------------------------------
    # Create an internal validation set
    # -----------------------------------------

    (
        X_internal_train,
        X_validation,
        y_internal_train,
        y_validation
    ) = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    print()
    print(
        "Internal training samples:",
        len(X_internal_train)
    )

    print(
        "Validation samples:",
        len(X_validation)
    )

    print(
        "Features available:",
        X_train.shape[1]
    )

    # -----------------------------------------
    # Fitness function
    # -----------------------------------------

    def fitness(mask):

        return evaluate_feature_subset(
            mask,
            X_internal_train,
            y_internal_train,
            X_validation,
            y_validation
        )

    # -----------------------------------------
    # Create RSA optimizer
    # -----------------------------------------

    selector = ReptileSearchFeatureSelector(
        population_size=10,
        iterations=15,
        random_state=42
    )

    # -----------------------------------------
    # Run RSA
    # -----------------------------------------

    best_mask = selector.optimize(
        fitness_function=fitness,
        number_of_features=X_train.shape[1]
    )

    selected_indices = np.where(
        best_mask == 1
    )[0]

    print()
    print("========================================")
    print("RSA RESULT")
    print("========================================")

    print(
        "Original features:",
        X_train.shape[1]
    )

    print(
        "Selected features:",
        len(selected_indices)
    )

    print(
        "Selected feature indices:"
    )

    print(selected_indices)

    # -----------------------------------------
    # Save RSA results
    # -----------------------------------------

    model_dir = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    rsa_data = {
        "selected_indices": selected_indices,
        "feature_mask": best_mask,
        "original_feature_count": X_train.shape[1],
        "fitness_history": selector.fitness_history
    }

    rsa_path = os.path.join(
        model_dir,
        "rsa_feature_selection.joblib"
    )

    joblib.dump(
        rsa_data,
        rsa_path
    )

    print()
    print(
        "RSA feature selection saved to:"
    )

    print(rsa_path)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        selected_indices
    )


if __name__ == "__main__":
    run_rsa()