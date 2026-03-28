from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "raw" / "train.csv"

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.")  

    df = pd.read_csv(INPUT_FILE)

    cabin_split = df["Cabin"].fillna("Unknown/0/Unknown").str.split("/", expand=True)

    df["CabinDeck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2]

    return df

if __name__ == "__main__":
    load_data()