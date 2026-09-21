import os
import joblib
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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


def main():

    print("========================================")
    print("ASL LANDMARK BASELINE MODEL")
    print("========================================")

    df = pd.read_csv(
        DATA_FILE
    )

    X = df.drop(
        "label",
        axis=1
    )

    y = df["label"]

    print()
    print("Total samples:", len(df))
    print("Features:", X.shape[1])
    print("Classes:", y.nunique())

    print()
    print("Class distribution:")
    print(
        y.value_counts().sort_index()
    )

    # -----------------------------------------
    # Train/test split
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

    print()
    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # -----------------------------------------
    # Train SVM
    # -----------------------------------------

    print()
    print("Training RBF-SVM...")

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        cache_size=4096
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("========================================")
    print("ASL LANDMARK BASELINE RESULTS")
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
        "asl_landmark_baseline_svm.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print()
    print(
        "Model saved to:"
    )

    print(
        model_path
    )


if __name__ == "__main__":
    main()