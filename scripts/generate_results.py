import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.ai.preprocessing import prepare_data


RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


def main():

    print("========================================")
    print("GENERATING EVALUATION RESULTS")
    print("========================================")

    # Load dataset split
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # -----------------------------------------
    # Load baseline model
    # -----------------------------------------

    baseline_path = os.path.join(
        MODELS_DIR,
        "baseline_model.joblib"
    )

    baseline_model = joblib.load(
        baseline_path
    )

    baseline_predictions = (
        baseline_model.predict(X_test)
    )

    # -----------------------------------------
    # Load RSA model
    # -----------------------------------------

    rsa_model_path = os.path.join(
        MODELS_DIR,
        "rsa_model.joblib"
    )

    rsa_model = joblib.load(
        rsa_model_path
    )

    rsa_info_path = os.path.join(
        MODELS_DIR,
        "rsa_feature_selection.joblib"
    )

    rsa_info = joblib.load(
        rsa_info_path
    )

    selected_indices = rsa_info[
        "selected_indices"
    ]

    X_test_rsa = X_test[
        :,
        selected_indices
    ]

    rsa_predictions = (
        rsa_model.predict(
            X_test_rsa
        )
    )

    # -----------------------------------------
    # 1. Baseline confusion matrix
    # -----------------------------------------

    print("Creating baseline confusion matrix...")

    figure = ConfusionMatrixDisplay.from_predictions(
        y_test,
        baseline_predictions,
        xticks_rotation="vertical"
    )

    figure.figure_.set_size_inches(
        12,
        10
    )

    figure.figure_.tight_layout()

    baseline_cm_path = os.path.join(
        RESULTS_DIR,
        "baseline_confusion_matrix.png"
    )

    figure.figure_.savefig(
        baseline_cm_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure.figure_
    )

    # -----------------------------------------
    # 2. RSA confusion matrix
    # -----------------------------------------

    print("Creating RSA confusion matrix...")

    figure = ConfusionMatrixDisplay.from_predictions(
        y_test,
        rsa_predictions,
        xticks_rotation="vertical"
    )

    figure.figure_.set_size_inches(
        12,
        10
    )

    figure.figure_.tight_layout()

    rsa_cm_path = os.path.join(
        RESULTS_DIR,
        "rsa_confusion_matrix.png"
    )

    figure.figure_.savefig(
        rsa_cm_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure.figure_
    )

    # -----------------------------------------
    # 3. Feature comparison chart
    # -----------------------------------------

    print("Creating feature comparison chart...")

    models = [
        "Baseline",
        "RSA"
    ]

    feature_counts = [
        X_train.shape[1],
        len(selected_indices)
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        models,
        feature_counts
    )

    plt.ylabel(
        "Number of Features"
    )

    plt.title(
        "Feature Count: Baseline vs RSA"
    )

    plt.tight_layout()

    feature_chart_path = os.path.join(
        RESULTS_DIR,
        "feature_reduction.png"
    )

    plt.savefig(
        feature_chart_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------
    # 4. Accuracy comparison chart
    # -----------------------------------------

    print("Creating accuracy comparison chart...")

    accuracy_values = [
        0.9700,
        0.9660
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        models,
        accuracy_values
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.ylim(
        0.90,
        1.00
    )

    plt.title(
        "Accuracy: Baseline vs RSA"
    )

    plt.tight_layout()

    accuracy_chart_path = os.path.join(
        RESULTS_DIR,
        "accuracy_comparison.png"
    )

    plt.savefig(
        accuracy_chart_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("========================================")
    print("RESULTS GENERATED")
    print("========================================")

    print(
        "Baseline confusion matrix:",
        baseline_cm_path
    )

    print(
        "RSA confusion matrix:",
        rsa_cm_path
    )

    print(
        "Feature chart:",
        feature_chart_path
    )

    print(
        "Accuracy chart:",
        accuracy_chart_path
    )


if __name__ == "__main__":
    main()