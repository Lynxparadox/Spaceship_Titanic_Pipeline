from pathlib import Path
import pickle

from sklearn.linear_model import LogisticRegression
from src.features.pre_processing import preprocess

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "logreg_model.pkl"


def train_model():

    X_train, X_test, y_train, y_test = preprocess()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model training successfully.")
    print(f"Model saved at: {MODEL_PATH}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()