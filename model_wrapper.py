import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import os

class ReadmissionPredictor:
    def __init__(self, model_uri=None, run_id=None):
        """
        Initialize the model wrapper.
        
        Args:
            model_uri (str): MLflow model URI. If None, uses run_id to fetch the model.
            run_id (str): MLflow run ID. Only used if model_uri is None.
        """
        if model_uri is None and run_id is None:
            raise ValueError("Either model_uri or run_id must be provided")
        
        if model_uri is None:
            model_uri = f"runs:/{run_id}/model"
        
        self.model = mlflow.xgboost.load_model(model_uri)
        self.feature_names = ['chol', 'crp', 'phos']
    
    def predict(self, data):
        """
        Make predictions using the loaded model.
        
        Args:
            data (dict or pd.DataFrame): Input data with features 'chol', 'crp', and 'phos'
            
        Returns:
            dict: Prediction results including probability and predicted class
        """
        # Convert input to DataFrame if it's a dictionary
        if isinstance(data, dict):
            # Handle both single input and batch inputs
            if all(isinstance(v, (int, float)) for v in data.values()):
                # Single input
                data = pd.DataFrame([data])
            else:
                # Batch input
                data = pd.DataFrame(data)
        
        # Ensure all required features are present
        missing_features = [col for col in self.feature_names if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Extract features in the correct order
        X = data[self.feature_names]
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        # Prepare results
        results = {
            'prediction': y_pred.tolist(),
            'probability': y_prob.tolist()
        }
        
        return results
    
    @classmethod
    def from_production(cls):
        """Load the production model from the model registry."""
        model_uri = "models:/xgboost/Production"
        return cls(model_uri=model_uri)
    
    @classmethod
    def from_latest(cls):
        """Load the latest version of the model from the model registry."""
        model_uri = "models:/xgboost/latest"
        return cls(model_uri=model_uri)

# Example usage
if __name__ == "__main__":
    # Example: Load model using run ID
    # predictor = ReadmissionPredictor(run_id="YOUR_RUN_ID")
    
    # Example: Load latest model version
    predictor = ReadmissionPredictor.from_latest()
    
    # Example: Make a prediction for a single patient
    patient_data = {
        'chol': 200,
        'crp': 3.5,
        'phos': 4.2
    }
    
    result = predictor.predict(patient_data)
    print(f"Prediction result: {result}")
    
    # Example: Batch prediction
    batch_data = {
        'chol': [200, 180, 220],
        'crp': [3.5, 2.8, 4.1],
        'phos': [4.2, 3.9, 4.5]
    }
    
    batch_result = predictor.predict(batch_data)
    print(f"Batch prediction results: {batch_result}")