from pathlib import Path
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.features.preprocessing import build_preprocessor
from src.data.data_ingestion import load_data

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"


def train_model():

    df = load_data()
    
    X = df.drop(columns=["Transported"], axis=1)
    y = df["Transported"].astype(int)

    preprocessor = build_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    model.fit(X, y)

    ARTIFACT_DIR = BASE_DIR / "artifacts"

    # save model
    with open(ARTIFACT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model training successfully.")
    print(f"Model saved!")

    return model

if __name__ == "__main__":
    train_model()