import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "models"
)

MODEL_DIR = os.path.abspath(MODEL_DIR)

os.makedirs(MODEL_DIR, exist_ok=True)


class SignLanguageModel:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Training completed.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):

        predictions = self.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print()
        print("===================================")
        print("BASELINE MODEL RESULTS")
        print("===================================")
        print(f"Accuracy: {accuracy:.4f}")
        print()
        print(
            classification_report(
                y_test,
                predictions
            )
        )

        return accuracy, predictions

    def save(self, filename="baseline_model.joblib"):

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        joblib.dump(
            self.model,
            path
        )

        print(f"Model saved to: {path}")

    def load(self, filename="baseline_model.joblib"):

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        self.model = joblib.load(path)

        print(f"Model loaded from: {path}")