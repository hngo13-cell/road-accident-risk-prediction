from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"

TARGET = "accident_risk"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# Load and split the labeled data
# --------------------------------------------------

def load_and_split_data():
    """
    Load train.csv and create training and validation sets.

    Returns:
        X_train, X_val, y_train, y_val
    """

    # Load labeled Kaggle training data
    train_df = pd.read_csv(TRAIN_PATH)

    # Separate predictors and target
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]

    # Create an 80/20 train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, y_train, y_val


# --------------------------------------------------
# Run verification when this file is executed directly
# --------------------------------------------------

if __name__ == "__main__":

    X_train, X_val, y_train, y_val = load_and_split_data()

    print("Train / validation split completed successfully.")

    print("\nSplit Shapes")
    print("------------")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:  ", y_val.shape)

    print("\nTotal Observations")
    print("------------------")
    print("Training:", len(X_train))
    print("Validation:", len(X_val))
    print("Total:", len(X_train) + len(X_val))

    print("\nTarget Distribution")
    print("-------------------")
    print(f"Training mean:   {y_train.mean():.6f}")
    print(f"Validation mean: {y_val.mean():.6f}")
    print(f"Training median: {y_train.median():.6f}")
    print(f"Validation median: {y_val.median():.6f}")