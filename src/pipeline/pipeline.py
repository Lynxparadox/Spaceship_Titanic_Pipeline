from src.data.loader_data import load_data
from src.models.train_model import train_model

def run_pipeline():
     
    print("Running data ingestion...")
    df = load_data()

    print("Training model...")
    model = train_model()

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()