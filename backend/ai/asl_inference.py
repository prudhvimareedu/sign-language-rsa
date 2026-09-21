import os
import sys
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.vision.feature_extractor import FeatureExtractor


class ASLInference:

    def __init__(self):

        model_path = os.path.join(
            PROJECT_ROOT,
            "backend",
            "models",
            "asl_rsa_svm.joblib"
        )

        indices_path = os.path.join(
            PROJECT_ROOT,
            "backend",
            "models",
            "asl_rsa_selected_indices.joblib"
        )

        self.model = joblib.load(model_path)
        self.selected_indices = joblib.load(indices_path)

        self.feature_extractor = FeatureExtractor()

        self.labels = sorted(
            self.model.classes_.tolist()
        )

        print("ASL RSA model loaded.")
        print(
            "Selected features:",
            len(self.selected_indices)
        )

    def predict(self, hand_landmarks):

        features = self.feature_extractor.extract(
            hand_landmarks
        )

        features = np.asarray(
            features,
            dtype=np.float32
        )

        selected_features = features[
            self.selected_indices
        ]

        selected_features = selected_features.reshape(
            1,
            -1
        )

        prediction = self.model.predict(
            selected_features
        )[0]

        decision_scores = self.model.decision_function(
            selected_features
        )

        if decision_scores.ndim == 1:
            confidence = float(
                np.max(decision_scores)
            )
        else:
            confidence = float(
                np.max(decision_scores[0])
            )

        return {
            "prediction": str(prediction),
            "confidence": confidence
        }