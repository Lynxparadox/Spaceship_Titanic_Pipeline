from pathlib import Path
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.features.preprocessing import build_preprocessor
from src.data.loader_data import load_data

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "pipeline_model.pkl"

def train_model():

    df = load_data()
    
    feature_columns = [
    "HomePlanet","CryoSleep","CabinDeck","CabinNum",
    "CabinSide","Destination","Age","VIP",
    "RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"
    ]

    X = df[feature_columns]
    y = df["Transported"].astype(int)

    preprocessor = build_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=5000))
        ]
    )

    model.fit(X, y)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # save model
    with open(ARTIFACT_DIR / "pipeline_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model training successfully.")
    print(f"Model saved at: {MODEL_PATH}!")

    return model

if __name__ == "__main__":
    train_model()
