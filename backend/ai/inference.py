import os
import joblib
import numpy as np
import pandas as pd


class SignLanguageInference:

    def __init__(self):

        project_root = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        models_dir = os.path.join(
            project_root,
            "models"
        )

        # Load RSA model
        self.model = joblib.load(
            os.path.join(
                models_dir,
                "rsa_model.joblib"
            )
        )

        # Load scaler
        self.scaler = joblib.load(
            os.path.join(
                models_dir,
                "feature_scaler.joblib"
            )
        )

        # Load RSA-selected feature indices
        rsa_data = joblib.load(
            os.path.join(
                models_dir,
                "rsa_feature_selection.joblib"
            )
        )

        self.selected_indices = np.array(
            rsa_data["selected_indices"]
        )

        # Original 63 feature names
        self.feature_names = []

        for i in range(21):
            self.feature_names.append(f"x_{i}")
            self.feature_names.append(f"y_{i}")
            self.feature_names.append(f"z_{i}")

        print(
            "Inference system loaded successfully."
        )

        print(
            "Selected features:",
            len(self.selected_indices)
        )

    def predict(self, features):

        features = np.asarray(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        # Convert to DataFrame with the same
        # feature names used during training
        features_df = pd.DataFrame(
            features,
            columns=self.feature_names
        )

        # Scale 63 original features
        scaled_features = self.scaler.transform(
            features_df
        )

        # Keep only RSA-selected features
        selected_features = (
            scaled_features[
                :,
                self.selected_indices
            ]
        )

        # Predict sign
        prediction = self.model.predict(
            selected_features
        )

        return prediction[0]