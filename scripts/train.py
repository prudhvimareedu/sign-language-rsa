import os
import sys
import joblib


# =========================================
# Add project root to Python path
# =========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# =========================================
# Project imports
# =========================================

from backend.ai.preprocessing import prepare_data
from backend.ai.model import SignLanguageModel


# =========================================
# Main training function
# =========================================

def main():

    print("===================================")
    print("SIGN LANGUAGE BASELINE TRAINING")
    print("===================================")

    # -----------------------------------------
    # Prepare dataset
    # -----------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = prepare_data()

    # -----------------------------------------
    # Create baseline model
    # -----------------------------------------

    model = SignLanguageModel()

    # -----------------------------------------
    # Train
    # -----------------------------------------

    model.train(
        X_train,
        y_train
    )

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------

    model.evaluate(
        X_test,
        y_test
    )

    # -----------------------------------------
    # Save model
    # -----------------------------------------

    model.save()

    # -----------------------------------------
    # Save scaler used for this exact training
    # -----------------------------------------

    models_dir = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models"
    )

    os.makedirs(
        models_dir,
        exist_ok=True
    )

    scaler_path = os.path.join(
        models_dir,
        "feature_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print()
    print(
        "Scaler saved to:"
    )

    print(
        scaler_path
    )

    print()
    print(
        "Baseline training finished successfully."
    )


# =========================================
# Entry point
# =========================================

if __name__ == "__main__":
    main()