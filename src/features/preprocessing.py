from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_preprocessor():

    categorical_features = [
        "HomePlanet",
        "CryoSleep",
        "CabinDeck",
        "CabinSide",
        "Destination",
        "VIP"
    ]

    numerical_features = [
        "CabinNum",
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck"
    ]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer([
            ("cat", categorical_pipeline, categorical_features),
            ("num", numerical_pipeline, numerical_features)
        ]
    )

    print("Preprocessing file created successfully.")

    return preprocessor

if __name__ == "__main__":
    build_preprocessor()