from spaceship_titanic_pipeline.data.data_ingestion import ingest_data 
from spaceship_titanic_pipeline.features.preprocessing import preprocess 
from spaceship_titanic_pipeline.models.train_model import train_model 
from evaluation import evaluate 

def run_pipeline_manual(): 
  print("Running data ingestion...") 
  df = ingest_data() 
  
  print("Running preprocessing...") 
  X_train, X_test, y_train, y_test, preprocessor = preprocess(df) 
  
  print("Training model...") 
  model = train_model(X_train, y_train, preprocessor) 
  
  print("Evaluating model...") 
  evaluate(model, X_test, y_test) 
  
  print("Pipeline completed successfully.") 
  
  if __name__ == "__main__":
    run_pipeline_manual()
