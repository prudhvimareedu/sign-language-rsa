import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "public",
    "sign_mnist_train.csv"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "public",
    "sign_mnist_test.csv"
)


def main():

    print("========================================")
    print("PUBLIC DATASET CHECK")
    print("========================================")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print()
    print("Training shape:", train_df.shape)
    print("Testing shape :", test_df.shape)

    print()
    print("Training columns:", len(train_df.columns))
    print("Testing columns :", len(test_df.columns))

    print()
    print("Training labels:")
    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Testing labels:")
    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Expected image pixels per sample:",
        len(train_df.columns) - 1
    )

    print()
    print("========================================")
    print("DATASET CHECK COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()