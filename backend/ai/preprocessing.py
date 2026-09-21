import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_FILE = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    ),
    "data",
    "processed",
    "sign_landmarks.csv"
)


def load_dataset():

    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Total samples:", len(df))
    print("Total columns:", len(df.columns))

    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())

    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y


def prepare_data():

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nData preparation completed.")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Number of features:", X_train.shape[1])

    return X_train, X_test, y_train, y_test, scaler