import os
import sys
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


from backend.ai.preprocessing import prepare_data


def main():

    print("========================================")
    print("RSA-SELECTED MODEL TRAINING")
    print("========================================")

    # -------------------------------------------------
    # Load the same train/test split
    # -------------------------------------------------

    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # -------------------------------------------------
    # Load RSA feature selection
    # -------------------------------------------------

    rsa_path = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models",
        "rsa_feature_selection.joblib"
    )

    rsa_data = joblib.load(rsa_path)

    selected_indices = rsa_data[
        "selected_indices"
    ]

    print()
    print(
        "Original features:",
        X_train.shape[1]
    )

    print(
        "RSA selected features:",
        len(selected_indices)
    )

    # -------------------------------------------------
    # Select RSA features
    # -------------------------------------------------

    X_train_selected = (
        X_train[:, selected_indices]
    )

    X_test_selected = (
        X_test[:, selected_indices]
    )

    # -------------------------------------------------
    # Train final RSA-selected model
    # -------------------------------------------------

    print()
    print(
        "Training RSA-selected "
        "Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train_selected,
        y_train
    )

    print("Training completed.")

    # -------------------------------------------------
    # Final evaluation
    # -------------------------------------------------

    predictions = model.predict(
        X_test_selected
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("========================================")
    print("RSA MODEL RESULTS")
    print("========================================")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # -------------------------------------------------
    # Save model
    # -------------------------------------------------

    model_path = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models",
        "rsa_model.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    # Save scaler too
    scaler_path = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models",
        "feature_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(
        "RSA model saved to:",
        model_path
    )

    print(
        "Scaler saved to:",
        scaler_path
    )

    print()
    print("RSA model training completed.")


if __name__ == "__main__":
    main()