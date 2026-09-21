import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def evaluate_feature_subset(
    mask,
    X_train,
    y_train,
    X_validation,
    y_validation
):
    """
    Evaluate one candidate feature subset.

    mask:
        1 = feature selected
        0 = feature not selected
    """

    selected_features = np.where(mask == 1)[0]

    # Reject extremely small feature sets
    if len(selected_features) < 3:
        return 1.0

    X_train_selected = X_train[:, selected_features]
    X_validation_selected = X_validation[:, selected_features]

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train_selected,
        y_train
    )

    predictions = model.predict(
        X_validation_selected
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    # Small penalty for using too many features
    feature_penalty = (
        0.02
        * len(selected_features)
        / X_train.shape[1]
    )

    fitness = (
        1.0
        - accuracy
        + feature_penalty
    )

    return fitness