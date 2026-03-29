from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

BASE_DIR    = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE  = BASE_DIR / "train.csv"
OUTPUT_FILE = INGESTED_DIR / "train.csv"


def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.") 

    df = pd.read_csv(INPUT_FILE)
    assert not df.empty, "Dataset is empty"

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data ingested: {INPUT_FILE} → {OUTPUT_FILE}")


if __name__ == "__main__":
    ingest_data()