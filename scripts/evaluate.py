import os
import sys
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Add project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.ai.preprocessing import prepare_data


def main():

    print("========================================")
    print("BASELINE vs RSA MODEL COMPARISON")
    print("========================================")

    # Load the exact same train/test split
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    models_dir = os.path.join(
        PROJECT_ROOT,
        "backend",
        "models"
    )

    # -----------------------------------------
    # Load baseline model
    # -----------------------------------------

    baseline_path = os.path.join(
        models_dir,
        "baseline_model.joblib"
    )

    baseline_model = joblib.load(
        baseline_path
    )

    baseline_predictions = (
        baseline_model.predict(X_test)
    )

    # -----------------------------------------
    # Load RSA information
    # -----------------------------------------

    rsa_info_path = os.path.join(
        models_dir,
        "rsa_feature_selection.joblib"
    )

    rsa_info = joblib.load(
        rsa_info_path
    )

    selected_indices = (
        rsa_info["selected_indices"]
    )

    # -----------------------------------------
    # Load RSA model
    # -----------------------------------------

    rsa_model_path = os.path.join(
        models_dir,
        "rsa_model.joblib"
    )

    rsa_model = joblib.load(
        rsa_model_path
    )

    X_test_rsa = X_test[
        :,
        selected_indices
    ]

    rsa_predictions = (
        rsa_model.predict(X_test_rsa)
    )

    # -----------------------------------------
    # Calculate metrics
    # -----------------------------------------

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    baseline_precision = precision_score(
        y_test,
        baseline_predictions,
        average="macro",
        zero_division=0
    )

    baseline_recall = recall_score(
        y_test,
        baseline_predictions,
        average="macro",
        zero_division=0
    )

    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        average="macro",
        zero_division=0
    )

    rsa_accuracy = accuracy_score(
        y_test,
        rsa_predictions
    )

    rsa_precision = precision_score(
        y_test,
        rsa_predictions,
        average="macro",
        zero_division=0
    )

    rsa_recall = recall_score(
        y_test,
        rsa_predictions,
        average="macro",
        zero_division=0
    )

    rsa_f1 = f1_score(
        y_test,
        rsa_predictions,
        average="macro",
        zero_division=0
    )

    # -----------------------------------------
    # Feature reduction
    # -----------------------------------------

    original_features = X_train.shape[1]

    selected_features = len(
        selected_indices
    )

    reduction = (
        (original_features - selected_features)
        / original_features
        * 100
    )

    # -----------------------------------------
    # Display results
    # -----------------------------------------

    print()
    print("========================================")
    print("FINAL COMPARISON")
    print("========================================")

    print()
    print("BASELINE RANDOM FOREST")
    print("----------------------------------------")
    print(
        f"Features : {original_features}"
    )
    print(
        f"Accuracy : {baseline_accuracy:.4f}"
    )
    print(
        f"Precision: {baseline_precision:.4f}"
    )
    print(
        f"Recall   : {baseline_recall:.4f}"
    )
    print(
        f"F1 Score : {baseline_f1:.4f}"
    )

    print()
    print("RSA + RANDOM FOREST")
    print("----------------------------------------")
    print(
        f"Features : {selected_features}"
    )
    print(
        f"Accuracy : {rsa_accuracy:.4f}"
    )
    print(
        f"Precision: {rsa_precision:.4f}"
    )
    print(
        f"Recall   : {rsa_recall:.4f}"
    )
    print(
        f"F1 Score : {rsa_f1:.4f}"
    )

    print()
    print("FEATURE REDUCTION")
    print("----------------------------------------")
    print(
        f"Original features : {original_features}"
    )
    print(
        f"Selected features : {selected_features}"
    )
    print(
        f"Reduction         : {reduction:.2f}%"
    )

    # -----------------------------------------
    # Save comparison results
    # -----------------------------------------

    results = {
        "baseline": {
            "features": original_features,
            "accuracy": baseline_accuracy,
            "precision": baseline_precision,
            "recall": baseline_recall,
            "f1": baseline_f1
        },
        "rsa": {
            "features": selected_features,
            "accuracy": rsa_accuracy,
            "precision": rsa_precision,
            "recall": rsa_recall,
            "f1": rsa_f1
        },
        "feature_reduction_percent": reduction
    }

    results_path = os.path.join(
        models_dir,
        "model_comparison.joblib"
    )

    joblib.dump(
        results,
        results_path
    )

    print()
    print(
        "Comparison saved to:"
    )
    print(results_path)


if __name__ == "__main__":
    main()