import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def evaluate_public_subset(
    mask,
    X_train,
    y_train,
    X_validation,
    y_validation
):
    """
    RSA fitness for HOG feature selection.

    A LinearSVC is used during the RSA search because
    hundreds of candidate subsets must be evaluated.
    The final selected subset will be tested later
    with the RBF-SVM baseline model.
    """

    selected_indices = np.where(mask == 1)[0]

    if len(selected_indices) < 10:
        return 1.0

    X_train_selected = X_train[
        :,
        selected_indices
    ]

    X_validation_selected = X_validation[
        :,
        selected_indices
    ]

    model = LinearSVC(
        C=1.0,
        dual="auto",
        max_iter=3000,
        random_state=42
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

    # Encourage feature reduction while
    # preserving classification performance.
    feature_penalty = (
        0.02
        * len(selected_indices)
        / X_train.shape[1]
    )

    fitness = (
        1.0
        - accuracy
        + feature_penalty
    )

    return fitness