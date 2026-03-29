from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ingested" / "train.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"

def split_cabin(df):

    cabin_split = df["Cabin"].fillna("Unknown/0/Unknown").str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2]

    return df

def preprocess():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    #split cabin into deck, num, side
    df = split_cabin(df)

    #13 features
    feature_columns = [
        "HomePlanet",
        "CryoSleep",
        "CabinDeck",
        "CabinNum",
        "CabinSide",
        "Destination",
        "Age",
        "VIP",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]

    target_column = "Transported"

    X = df[feature_columns].copy()
    y = df[target_column].astype(int)

    #fill missing numerical values
    numeric_cols = [
        "CabinNum",
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    #categorical columns
    categorical_cols = [
        "HomePlanet",
        "CryoSleep",
        "CabinDeck",
        "CabinSide",
        "Destination",
        "VIP",
    ]

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    #save encoders
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(encoders, f)

    #split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Preprocessing completed.")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess()